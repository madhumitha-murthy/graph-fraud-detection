"""
Hadoop HDFS utilities for the Elliptic fraud detection pipeline.

In production fraud detection systems (TikTok BRIC, PayPal, Mastercard),
raw transaction graphs are stored on HDFS and read directly by Spark jobs
running on YARN clusters. This module provides:

    1. HDFS path resolution  — local ↔ HDFS transparent switching
    2. HDFS directory listing — via Spark's Hadoop FileSystem API
    3. HDFS write / read     — Parquet output to HDFS
    4. Hadoop config helpers — expose cluster config from SparkSession

Usage (local mode — Spark uses local filesystem as HDFS substitute):
    spark = build_spark_session()
    hfs   = HadoopFS(spark)
    hfs.ls("data/elliptic")
    hfs.put("data/elliptic/enriched.parquet", "hdfs:///fraud/enriched.parquet")

Usage (cluster mode — swap HDFS_NAMENODE in .env):
    export HDFS_NAMENODE=hdfs://namenode:9000
    python -m src.features.spark_feature_engineering --hdfs
"""

import os
from pathlib import Path
from typing import Optional

from pyspark.sql import SparkSession, DataFrame


HDFS_NAMENODE = os.getenv("HDFS_NAMENODE", "")   # empty → local mode


def hdfs_path(relative_path: str) -> str:
    """
    Return a fully qualified path — HDFS URI in cluster mode, local in dev.

    Examples:
        hdfs_path("fraud/features")
        → "hdfs://namenode:9000/fraud/features"  (cluster)
        → "data/elliptic/features"               (local dev)
    """
    if HDFS_NAMENODE:
        return f"{HDFS_NAMENODE.rstrip('/')}/{relative_path.lstrip('/')}"
    return relative_path


class HadoopFS:
    """
    Thin wrapper around Spark's Hadoop FileSystem Java API.

    Spark ships with Hadoop binaries and exposes the full
    org.apache.hadoop.fs.FileSystem API via the JVM gateway.
    This class makes that API usable from Python without a standalone
    Hadoop installation.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self._jvm  = spark.sparkContext._jvm
        self._conf = spark.sparkContext._jsc.hadoopConfiguration()

    def _get_fs(self, path: str):
        """Return Hadoop FileSystem object for the given path URI."""
        uri  = self._jvm.java.net.URI(path if "://" in path else f"file:///{Path(path).resolve()}")
        return self._jvm.org.apache.hadoop.fs.FileSystem.get(uri, self._conf)

    def ls(self, path: str) -> list[dict]:
        """
        List files/directories at path — equivalent to `hadoop fs -ls`.

        Returns list of dicts with: name, size, is_dir, modification_time.
        """
        resolved = hdfs_path(path)
        try:
            fs       = self._get_fs(resolved)
            hpath    = self._jvm.org.apache.hadoop.fs.Path(resolved)
            statuses = fs.listStatus(hpath)
            results  = []
            for s in statuses:
                results.append({
                    "name":              s.getPath().getName(),
                    "full_path":         str(s.getPath()),
                    "size_bytes":        s.getLen(),
                    "is_dir":            s.isDirectory(),
                    "modification_time": s.getModificationTime(),
                })
            return results
        except Exception as e:
            print(f"HDFS ls failed for {resolved}: {e}")
            return []

    def exists(self, path: str) -> bool:
        """Check if path exists — equivalent to `hadoop fs -test -e`."""
        resolved = hdfs_path(path)
        try:
            fs    = self._get_fs(resolved)
            hpath = self._jvm.org.apache.hadoop.fs.Path(resolved)
            return bool(fs.exists(hpath))
        except Exception:
            return False

    def mkdir(self, path: str) -> None:
        """Create directory — equivalent to `hadoop fs -mkdir -p`."""
        resolved = hdfs_path(path)
        fs       = self._get_fs(resolved)
        hpath    = self._jvm.org.apache.hadoop.fs.Path(resolved)
        fs.mkdirs(hpath)
        print(f"HDFS mkdir: {resolved}")

    def rm(self, path: str, recursive: bool = True) -> None:
        """Delete path — equivalent to `hadoop fs -rm -r`."""
        resolved = hdfs_path(path)
        fs       = self._get_fs(resolved)
        hpath    = self._jvm.org.apache.hadoop.fs.Path(resolved)
        fs.delete(hpath, recursive)
        print(f"HDFS rm: {resolved}")

    def write_parquet(self, df: DataFrame, path: str, mode: str = "overwrite") -> str:
        """
        Write Spark DataFrame to Parquet on HDFS (or local in dev mode).

        In cluster mode: df.write.parquet("hdfs://namenode:9000/fraud/features")
        In local mode:   df.write.parquet("data/elliptic/features.parquet")
        """
        resolved = hdfs_path(path)
        df.write.mode(mode).parquet(resolved)
        print(f"Written {df.count():,} rows → {resolved}")
        return resolved

    def read_parquet(self, path: str) -> DataFrame:
        """Read Parquet from HDFS (or local) back into Spark DataFrame."""
        resolved = hdfs_path(path)
        df = self.spark.read.parquet(resolved)
        print(f"Read {df.count():,} rows ← {resolved}")
        return df

    def get_hadoop_config(self) -> dict:
        """
        Expose key Hadoop configuration values from SparkSession.
        Useful for debugging cluster connectivity and HDFS settings.
        """
        keys = [
            "fs.defaultFS",
            "dfs.replication",
            "mapreduce.framework.name",
            "yarn.resourcemanager.address",
            "hadoop.tmp.dir",
        ]
        config = {}
        for k in keys:
            val = self._conf.get(k)
            config[k] = val if val else "(not set)"
        return config

    def print_hadoop_config(self) -> None:
        """Print Hadoop cluster configuration — useful for interview demos."""
        config = self.get_hadoop_config()
        print("\nHadoop Configuration:")
        print("-" * 45)
        for k, v in config.items():
            print(f"  {k:<38} {v}")
        mode = "CLUSTER" if HDFS_NAMENODE else "LOCAL (dev)"
        print(f"  {'Mode':<38} {mode}")
        print("-" * 45)


def run_hdfs_demo(spark: SparkSession, data_dir: str = "data/elliptic") -> None:
    """
    Demonstrate full HDFS read/write cycle using Spark's Hadoop FileSystem API.

    This is what the same code does on a real Hadoop cluster:
        hadoop fs -ls /fraud/elliptic/
        hadoop fs -mkdir /fraud/enriched/
        (Spark writes Parquet to HDFS, reads back, validates row count)
    """
    hfs = HadoopFS(spark)

    print("\n── Hadoop FileSystem Demo ──────────────────────────────")
    hfs.print_hadoop_config()

    print(f"\nListing {data_dir}:")
    files = hfs.ls(data_dir)
    for f in files:
        size_mb = f["size_bytes"] / 1e6
        marker  = "[DIR]" if f["is_dir"] else f"[{size_mb:.1f} MB]"
        print(f"  {marker:>12}  {f['name']}")

    enriched_path = f"{data_dir}/elliptic_enriched_features.parquet"
    if hfs.exists(enriched_path):
        print(f"\nReading enriched features from: {enriched_path}")
        df = hfs.read_parquet(enriched_path)
        print(f"Schema: {len(df.columns)} columns")
        df.select("txid", "time_step", "out_degree", "in_degree",
                  "neighbour_illicit_ratio").show(5, truncate=False)

    print("── HDFS Demo complete ──────────────────────────────────\n")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.features.spark_feature_engineering import build_spark_session

    spark = build_spark_session("HadoopFSDemo")
    spark.sparkContext.setLogLevel("WARN")
    run_hdfs_demo(spark)
    spark.stop()
