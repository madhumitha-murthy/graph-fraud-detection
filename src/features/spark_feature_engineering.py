"""
PySpark Feature Engineering — Elliptic Bitcoin Transaction Graph.

Computes graph-structural features at scale using PySpark DataFrames:
    - In-degree  / out-degree per node (transaction fan-in / fan-out)
    - Degree centrality (normalised degree)
    - Neighbour illicit ratio (fraction of known-illicit neighbours)
    - Temporal transaction velocity (tx count per node per time step)
    - Standardised feature vector ready for GNN input

Why PySpark here:
    The Elliptic graph has 234k edges and 203k nodes. Computing
    neighbour aggregations with pandas requires full in-memory joins.
    PySpark distributes these across partitions and scales naturally
    to production graphs with billions of edges (e.g. TikTok social graph).

Output:
    data/elliptic/elliptic_enriched_features.parquet
    — drop-in replacement for the raw features CSV, with 6 new columns appended.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_DIR

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import (
        StructType, StructField, StringType, FloatType, IntegerType, LongType
    )
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml import Pipeline
except ImportError:
    raise ImportError("Install PySpark: pip install pyspark")


def build_spark_session(app_name: str = "EllipticFraudFeatures") -> "SparkSession":
    # Java 17+ requires explicit module opens for Spark internals
    java_opts = " ".join([
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
        "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
        "--add-opens=java.base/java.io=ALL-UNNAMED",
        "--add-opens=java.base/java.net=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/java.util=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
        "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    ])
    return (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory",            "4g")
        .config("spark.sql.shuffle.partitions",   "8")
        .config("spark.ui.showConsoleProgress",   "false")
        .config("spark.driver.extraJavaOptions",  java_opts)
        .config("spark.executor.extraJavaOptions", java_opts)
        .getOrCreate()
    )


def load_raw_data(spark: "SparkSession") -> tuple:
    """Load Elliptic CSVs into Spark DataFrames."""

    # ── Features ─────────────────────────────────────────────────────────────
    feat_path = str(DATA_DIR / "elliptic_txs_features.csv")
    n_feat_cols = 165
    feat_schema = StructType(
        [StructField("txid",      LongType(),  True),
         StructField("time_step", IntegerType(), True)]
        + [StructField(f"f{i}", FloatType(), True) for i in range(n_feat_cols)]
    )
    features_df = spark.read.csv(feat_path, schema=feat_schema, header=False)

    # ── Edges ─────────────────────────────────────────────────────────────────
    edge_path = str(DATA_DIR / "elliptic_txs_edgelist.csv")
    edges_df = spark.read.csv(edge_path, header=True, inferSchema=True)
    edges_df = edges_df.toDF("src", "dst")

    # ── Labels ────────────────────────────────────────────────────────────────
    class_path = str(DATA_DIR / "elliptic_txs_classes.csv")
    classes_df = spark.read.csv(class_path, header=True, inferSchema=True)
    classes_df = classes_df.toDF("txid", "label")
    # Normalise: "1" → illicit, "2" → licit, else unknown
    classes_df = classes_df.withColumn(
        "is_illicit",
        F.when(F.col("label").cast("string") == "1", 1)
         .when(F.col("label").cast("string") == "2", 0)
         .otherwise(-1)
    )

    return features_df, edges_df, classes_df


def compute_degree_features(
    features_df: "DataFrame",
    edges_df:    "DataFrame",
) -> "DataFrame":
    """
    Compute per-node degree statistics.

    out_degree : number of outgoing edges (how many txs this node funds)
    in_degree  : number of incoming edges (how many txs fund this node)
    total_degree : sum
    degree_centrality : total_degree / (N-1), normalised

    High out-degree with many illicit destinations → likely spam/bot behaviour.
    High in-degree from many sources → potential aggregation node (money mule).
    """
    n_nodes = features_df.count()

    out_deg = (
        edges_df.groupBy("src")
        .agg(F.count("dst").alias("out_degree"))
        .withColumnRenamed("src", "txid")
    )
    in_deg = (
        edges_df.groupBy("dst")
        .agg(F.count("src").alias("in_degree"))
        .withColumnRenamed("dst", "txid")
    )

    degree_df = (
        features_df.select("txid")
        .join(out_deg, on="txid", how="left")
        .join(in_deg,  on="txid", how="left")
        .fillna(0, subset=["out_degree", "in_degree"])
        .withColumn("total_degree",       F.col("out_degree") + F.col("in_degree"))
        .withColumn("degree_centrality",  F.col("total_degree") / (n_nodes - 1))
    )
    return degree_df


def compute_neighbour_illicit_ratio(
    features_df: "DataFrame",
    edges_df:    "DataFrame",
    classes_df:  "DataFrame",
) -> "DataFrame":
    """
    For each node, compute the fraction of its direct neighbours that are illicit.

    neighbour_illicit_ratio: illicit_neighbours / total_labelled_neighbours

    This is a first-order guilt-by-association signal — a core feature in
    production fraud detection systems (TikTok BRIC, PayPal, Mastercard).
    Nodes with high ratios are likely part of a fraud ring even if their
    own label is unknown.
    """
    # Join edges with labels of destination nodes
    labelled = classes_df.filter(F.col("is_illicit") >= 0).select("txid", "is_illicit")

    edge_with_label = (
        edges_df
        .join(labelled.withColumnRenamed("txid", "dst")
                      .withColumnRenamed("is_illicit", "dst_illicit"),
              on="dst", how="left")
    )

    neighbour_stats = (
        edge_with_label
        .groupBy("src")
        .agg(
            F.sum(F.when(F.col("dst_illicit") == 1, 1).otherwise(0))
             .alias("illicit_neighbours"),
            F.count(F.when(F.col("dst_illicit") >= 0, 1))
             .alias("labelled_neighbours"),
        )
        .withColumn(
            "neighbour_illicit_ratio",
            F.when(F.col("labelled_neighbours") > 0,
                   F.col("illicit_neighbours") / F.col("labelled_neighbours"))
             .otherwise(0.0)
        )
        .withColumnRenamed("src", "txid")
        .select("txid", "illicit_neighbours", "labelled_neighbours", "neighbour_illicit_ratio")
    )
    return neighbour_stats


def compute_temporal_velocity(features_df: "DataFrame") -> "DataFrame":
    """
    Transaction velocity: how many transactions occur at the same time step.

    High velocity at a single time step from a node's neighbourhood
    is a signal for coordinated inauthentic behaviour (bot bursts).
    """
    velocity = (
        features_df
        .groupBy("time_step")
        .agg(F.count("txid").alias("timestep_tx_count"))
    )
    return features_df.select("txid", "time_step").join(velocity, on="time_step", how="left")


def enrich_features(spark: "SparkSession") -> "DataFrame":
    """
    Full enrichment pipeline:
        raw features + degree stats + neighbour illicit ratio + velocity
        → standardised Parquet output
    """
    print("Loading raw data...")
    features_df, edges_df, classes_df = load_raw_data(spark)

    print(f"  Nodes: {features_df.count():,}  Edges: {edges_df.count():,}")

    # ── Compute derived features ──────────────────────────────────────────────
    print("Computing degree features...")
    degree_df = compute_degree_features(features_df, edges_df)

    print("Computing neighbour illicit ratio...")
    neighbour_df = compute_neighbour_illicit_ratio(features_df, edges_df, classes_df)

    print("Computing temporal velocity...")
    velocity_df = compute_temporal_velocity(features_df)

    # ── Join everything ───────────────────────────────────────────────────────
    print("Joining enriched features...")
    enriched = (
        features_df
        .join(degree_df,     on="txid", how="left")
        .join(neighbour_df,  on="txid", how="left")
        .join(velocity_df.select("txid", "timestep_tx_count"), on="txid", how="left")
        .fillna(0.0, subset=[
            "out_degree", "in_degree", "total_degree", "degree_centrality",
            "illicit_neighbours", "labelled_neighbours", "neighbour_illicit_ratio",
            "timestep_tx_count",
        ])
    )

    # ── Standardise new columns with Spark ML Pipeline ───────────────────────
    print("Standardising new features with Spark ML Pipeline...")
    new_cols = [
        "out_degree", "in_degree", "total_degree", "degree_centrality",
        "neighbour_illicit_ratio", "timestep_tx_count",
    ]
    assembler = VectorAssembler(inputCols=new_cols, outputCol="new_features_vec")
    scaler    = StandardScaler(
        inputCol="new_features_vec",
        outputCol="new_features_scaled",
        withMean=True, withStd=True,
    )
    pipeline  = Pipeline(stages=[assembler, scaler])
    model     = pipeline.fit(enriched)
    enriched  = model.transform(enriched).drop("new_features_vec")

    # ── Write output ─────────────────────────────────────────────────────────
    out_path = str(DATA_DIR / "elliptic_enriched_features.parquet")
    print(f"Writing enriched features to {out_path} ...")
    (
        enriched
        .repartition(4)
        .write.mode("overwrite")
        .parquet(out_path)
    )

    print("\nEnrichment complete.")
    print(f"  Original features : 165")
    print(f"  New features added: {len(new_cols)} (degree, centrality, neighbour ratio, velocity)")
    print(f"  Output            : {out_path}")

    enriched.select(
        "txid", "time_step",
        "out_degree", "in_degree", "degree_centrality",
        "neighbour_illicit_ratio", "timestep_tx_count",
    ).show(10, truncate=False)

    return enriched


def load_enriched_for_pytorch(spark: "SparkSession") -> "np.ndarray":
    """
    Load the enriched Parquet back as a numpy array for PyTorch GNN training.
    Appends the 6 new standardised features to the original 165-column feature matrix.
    """
    import numpy as np
    from pyspark.ml.functions import vector_to_array

    out_path = str(DATA_DIR / "elliptic_enriched_features.parquet")
    df = spark.read.parquet(out_path)

    # Convert scaled vector column to individual float columns
    df = df.withColumn("scaled_arr", vector_to_array("new_features_scaled"))
    for i, col_name in enumerate([
        "sc_out_deg", "sc_in_deg", "sc_total_deg",
        "sc_centrality", "sc_illicit_ratio", "sc_velocity"
    ]):
        df = df.withColumn(col_name, F.col("scaled_arr")[i])

    # Collect ordered by txid for consistent node indexing
    df_sorted = df.orderBy("txid")
    pandas_df  = df_sorted.select(
        "txid", "time_step",
        *[f"f{i}" for i in range(165)],
        "sc_out_deg", "sc_in_deg", "sc_total_deg",
        "sc_centrality", "sc_illicit_ratio", "sc_velocity",
    ).toPandas()

    feat_cols = [f"f{i}" for i in range(165)] + [
        "sc_out_deg", "sc_in_deg", "sc_total_deg",
        "sc_centrality", "sc_illicit_ratio", "sc_velocity",
    ]
    return pandas_df["txid"].values, pandas_df[feat_cols].values.astype("float32")


if __name__ == "__main__":
    spark = build_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    enrich_features(spark)
    spark.stop()
    print("Done.")
