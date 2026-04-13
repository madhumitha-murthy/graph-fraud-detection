"""
Load the Elliptic fraud graph into Neo4j for production-style graph queries.

Neo4j enables Cypher queries that mirror real fraud detection systems:
  - Find fraud clusters (connected components of illicit nodes)
  - Identify bridge nodes (licit nodes connected to illicit clusters)
  - Trace money flow paths between flagged transactions

Usage:
    loader = Neo4jFraudLoader()
    loader.load_graph(data, meta, anomaly_scores)
    loader.close()
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jFraudLoader:
    """Loads fraud graph into Neo4j and exposes Cypher-based fraud queries."""

    def __init__(self):
        from neo4j import GraphDatabase
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self._verify_connection()

    def _verify_connection(self):
        with self.driver.session() as session:
            session.run("RETURN 1")
        print(f"Neo4j connected: {NEO4J_URI}")

    def close(self):
        self.driver.close()

    # ── Schema ───────────────────────────────────────────────────────────────

    def create_constraints(self):
        """Create uniqueness constraints for fast lookups."""
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT tx_id_unique IF NOT EXISTS "
                "FOR (t:Transaction) REQUIRE t.tx_id IS UNIQUE"
            )

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Neo4j graph cleared.")

    # ── Bulk load ────────────────────────────────────────────────────────────

    def load_graph(
        self,
        data: Data,
        meta: dict,
        anomaly_scores: Optional[np.ndarray] = None,
        batch_size: int = 2000,
    ):
        """
        Load all Transaction nodes and SENDS edges into Neo4j.

        Node properties:
            tx_id, time_step, label (illicit/licit/unknown),
            anomaly_score (from GAE reconstruction error)
        """
        self.create_constraints()

        tx_ids     = data.tx_id.numpy()
        time_steps = data.time_step.numpy()
        labels     = data.y.numpy()
        label_map  = {1: "illicit", 0: "licit", -1: "unknown"}

        if anomaly_scores is None:
            anomaly_scores = np.zeros(len(tx_ids))

        # ── Nodes ────────────────────────────────────────────────────────────
        print("Loading Transaction nodes...")
        nodes = [
            {
                "tx_id":         int(tx_ids[i]),
                "time_step":     int(time_steps[i]),
                "label":         label_map.get(int(labels[i]), "unknown"),
                "is_illicit":    int(labels[i]) == 1,
                "anomaly_score": float(anomaly_scores[i]),
            }
            for i in range(len(tx_ids))
        ]

        with self.driver.session() as session:
            for i in tqdm(range(0, len(nodes), batch_size), desc="Nodes"):
                batch = nodes[i : i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (t:Transaction {tx_id: row.tx_id})
                    SET t.time_step     = row.time_step,
                        t.label         = row.label,
                        t.is_illicit    = row.is_illicit,
                        t.anomaly_score = row.anomaly_score
                    """,
                    batch=batch,
                )

        # ── Edges ────────────────────────────────────────────────────────────
        print("Loading SENDS edges...")
        src_arr, dst_arr = data.edge_index.numpy()
        idx_to_tx = {v: int(k) for k, v in meta["tx_to_idx"].items()}

        edges = [
            {"src": idx_to_tx.get(int(src_arr[i]), -1),
             "dst": idx_to_tx.get(int(dst_arr[i]), -1)}
            for i in range(0, len(src_arr), 2)      # skip reverse edges
            if idx_to_tx.get(int(src_arr[i])) is not None
        ]

        with self.driver.session() as session:
            for i in tqdm(range(0, len(edges), batch_size), desc="Edges"):
                batch = edges[i : i + batch_size]
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (a:Transaction {tx_id: row.src})
                    MATCH (b:Transaction {tx_id: row.dst})
                    MERGE (a)-[:SENDS]->(b)
                    """,
                    batch=batch,
                )

        print(f"Neo4j loaded: {len(nodes):,} nodes, {len(edges):,} edges")

    # ── Fraud Queries (Cypher) ────────────────────────────────────────────────

    def get_fraud_clusters(self, min_size: int = 3) -> list[dict]:
        """
        Find connected subgraphs where ALL nodes are illicit.
        These represent coordinated fraud rings.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction {is_illicit: true})
                WITH collect(t.tx_id) AS illicit_ids
                CALL {
                    WITH illicit_ids
                    MATCH (a:Transaction)-[:SENDS]->(b:Transaction)
                    WHERE a.tx_id IN illicit_ids AND b.tx_id IN illicit_ids
                    RETURN a.tx_id AS src, b.tx_id AS dst
                }
                RETURN src, dst
                """
            )
            return [dict(r) for r in result]

    def get_bridge_nodes(self, anomaly_threshold: float = 0.7) -> list[dict]:
        """
        Find licit nodes with high anomaly scores that connect to illicit nodes.
        These are likely money-laundering intermediaries.
        """
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (licit:Transaction {is_illicit: false})
                WHERE licit.anomaly_score >= $threshold
                MATCH (licit)-[:SENDS]->(illicit:Transaction {is_illicit: true})
                RETURN licit.tx_id   AS bridge_tx,
                       licit.anomaly_score AS score,
                       collect(illicit.tx_id) AS connected_illicit_txs
                ORDER BY licit.anomaly_score DESC
                LIMIT 50
                """,
                threshold=anomaly_threshold,
            )
            return [dict(r) for r in result]

    def get_high_risk_subgraph(self, tx_id: int, hops: int = 2) -> list[dict]:
        """
        Return the k-hop neighbourhood of a given transaction for risk tracing.
        Used by the API's /explain endpoint.
        """
        with self.driver.session() as session:
            result = session.run(
                f"""
                MATCH path = (start:Transaction {{tx_id: $tx_id}})-[:SENDS*1..{hops}]-(neighbour)
                RETURN DISTINCT
                    neighbour.tx_id      AS tx_id,
                    neighbour.label      AS label,
                    neighbour.anomaly_score AS anomaly_score,
                    length(path)         AS hops_from_target
                ORDER BY hops_from_target, neighbour.anomaly_score DESC
                """,
                tx_id=tx_id,
            )
            return [dict(r) for r in result]

    def get_temporal_fraud_stats(self) -> list[dict]:
        """Fraud rate per time step — used for drift monitoring."""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (t:Transaction)
                WHERE t.label IN ['illicit', 'licit']
                WITH t.time_step AS ts,
                     sum(CASE WHEN t.is_illicit THEN 1 ELSE 0 END) AS n_illicit,
                     count(t) AS n_total
                RETURN ts,
                       n_illicit,
                       n_total,
                       toFloat(n_illicit) / n_total AS fraud_rate
                ORDER BY ts
                """
            )
            return [dict(r) for r in result]
