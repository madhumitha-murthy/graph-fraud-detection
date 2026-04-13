"""
Graph Fraud Detection API — FastAPI

Endpoints:
    GET  /health                   — liveness check
    POST /score                    — score a single transaction node
    POST /score/batch              — score multiple transaction nodes
    GET  /explain/{tx_id}          — k-hop neighbourhood from Neo4j
    GET  /drift/status             — anomaly score drift report
    GET  /fraud-clusters           — top fraud rings from Neo4j
    GET  /bridge-nodes             — high-risk licit nodes bridging fraud clusters

All scoring uses an ensemble of:
    - GAE reconstruction error  (unsupervised, 40% weight)
    - GraphSAGE fraud probability (supervised,  60% weight)
"""

import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    MODELS_DIR, GAE_HIDDEN_DIM, GAE_LATENT_DIM,
    SAGE_HIDDEN_DIM, SAGE_NUM_LAYERS, SAGE_DROPOUT,
    GAE_ANOMALY_PCTILE,
)
from src.api.schemas import (
    ScoreRequest, BatchScoreRequest, FraudScore, BatchScoreResponse,
    ExplainResponse, DriftStatusResponse, HealthResponse,
)
from src.models.graph_autoencoder import GraphAutoencoder
from src.models.graphsage import FraudGraphSAGE
from src.evaluation.drift_monitor import detect_score_drift

# ── App state ─────────────────────────────────────────────────────────────────

class AppState:
    gae_model:      Optional[GraphAutoencoder] = None
    sage_model:     Optional[FraudGraphSAGE]   = None
    anomaly_scores: Optional[np.ndarray]       = None
    threshold:      float                      = 0.5
    tx_to_idx:      dict                       = {}
    time_steps:     Optional[np.ndarray]       = None
    labels:         Optional[np.ndarray]       = None
    neo4j_loader    = None
    neo4j_ok:       bool                       = False
    n_nodes:        int                        = 0
    data            = None


state = AppState()


def _risk_label(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.65: return "high"
    if score >= 0.40: return "medium"
    return "low"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and data on startup."""
    device = torch.device("cpu")

    # ── Load GAE ──────────────────────────────────────────────────────────────
    gae_path = MODELS_DIR / "gae_model.pt"
    if gae_path.exists():
        try:
            from src.graph.builder import load_elliptic
            data, meta = load_elliptic()
            state.data       = data
            state.tx_to_idx  = meta["tx_to_idx"]
            state.time_steps = data.time_step.numpy()
            state.labels     = data.y.numpy()
            state.n_nodes    = meta["n_nodes"]

            gae = GraphAutoencoder(data.num_node_features, GAE_HIDDEN_DIM, GAE_LATENT_DIM)
            gae.load_state_dict(torch.load(gae_path, map_location=device))
            gae.eval()
            state.gae_model = gae

            scores_path = MODELS_DIR / "gae_anomaly_scores.npy"
            if scores_path.exists():
                state.anomaly_scores = np.load(scores_path)
                state.threshold = float(
                    np.percentile(state.anomaly_scores, GAE_ANOMALY_PCTILE)
                )
            print(f"GAE loaded | threshold={state.threshold:.4f}")
        except Exception as e:
            print(f"GAE load failed: {e}")
    else:
        print(f"GAE model not found at {gae_path} — train first.")

    # ── Load GraphSAGE ────────────────────────────────────────────────────────
    sage_path = MODELS_DIR / "graphsage_model.pt"
    if sage_path.exists() and state.data is not None:
        try:
            sage = FraudGraphSAGE(
                state.data.num_node_features, SAGE_HIDDEN_DIM,
                SAGE_NUM_LAYERS, SAGE_DROPOUT,
            )
            sage.load_state_dict(torch.load(sage_path, map_location=device))
            sage.eval()
            state.sage_model = sage
            print("GraphSAGE loaded")
        except Exception as e:
            print(f"GraphSAGE load failed: {e}")

    # ── Connect Neo4j ─────────────────────────────────────────────────────────
    try:
        from src.graph.neo4j_loader import Neo4jFraudLoader
        state.neo4j_loader = Neo4jFraudLoader()
        state.neo4j_ok     = True
        print("Neo4j connected")
    except Exception as e:
        print(f"Neo4j not available: {e}")

    yield

    if state.neo4j_loader:
        state.neo4j_loader.close()


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title        = "Graph Fraud Detection API",
    description  = "Unsupervised + supervised fraud detection on Bitcoin transaction graphs using GAE + GraphSAGE + Neo4j",
    version      = "1.0.0",
    lifespan     = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_node_idx(tx_id: int) -> int:
    idx = state.tx_to_idx.get(tx_id)
    if idx is None:
        raise HTTPException(status_code=404, detail=f"tx_id {tx_id} not found in graph")
    return idx


def _ensemble_score(gae_score: float, sage_prob: float) -> float:
    return 0.4 * gae_score + 0.6 * sage_prob


def _score_node(tx_id: int) -> FraudScore:
    idx = _get_node_idx(tx_id)

    # GAE score (precomputed)
    gae_s = float(state.anomaly_scores[idx]) if state.anomaly_scores is not None else 0.0
    gae_norm = min(gae_s / (state.threshold + 1e-8), 1.0)

    # GraphSAGE probability
    sage_p = 0.5  # default if model not loaded
    if state.sage_model is not None and state.data is not None:
        with torch.no_grad():
            sage_p = float(
                state.sage_model.predict_proba(state.data.x, state.data.edge_index)[idx]
            )

    ensemble = _ensemble_score(gae_norm, sage_p)
    return FraudScore(
        tx_id          = tx_id,
        gae_score      = round(gae_norm, 4),
        sage_prob      = round(sage_p,   4),
        ensemble_score = round(ensemble, 4),
        risk_label     = _risk_label(ensemble),
        is_flagged     = ensemble >= 0.5,
    )


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status           = "ok",
        gae_loaded       = state.gae_model is not None,
        sage_loaded      = state.sage_model is not None,
        neo4j_connected  = state.neo4j_ok,
        n_nodes_indexed  = state.n_nodes,
    )


@app.post("/score", response_model=FraudScore)
def score_single(req: ScoreRequest):
    """Score a single transaction node."""
    if state.anomaly_scores is None:
        raise HTTPException(503, "Models not loaded")
    return _score_node(req.tx_id)


@app.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(req: BatchScoreRequest):
    """Score multiple transaction nodes in one request."""
    if state.anomaly_scores is None:
        raise HTTPException(503, "Models not loaded")
    results = [_score_node(tx_id) for tx_id in req.tx_ids]
    flagged = [r for r in results if r.is_flagged]
    return BatchScoreResponse(
        results    = results,
        n_flagged  = len(flagged),
        flag_rate  = len(flagged) / max(len(results), 1),
    )


@app.get("/explain/{tx_id}", response_model=ExplainResponse)
def explain(tx_id: int, hops: int = 2):
    """
    Return k-hop neighbourhood of a transaction from Neo4j.
    Identifies how many neighbours are illicit (fraud ring context).
    """
    if not state.neo4j_ok:
        raise HTTPException(503, "Neo4j not connected")

    neighbourhood = state.neo4j_loader.get_high_risk_subgraph(tx_id, hops=hops)
    n_illicit  = sum(1 for n in neighbourhood if n.get("label") == "illicit")
    n_licit    = sum(1 for n in neighbourhood if n.get("label") == "licit")

    score_obj  = _score_node(tx_id) if state.anomaly_scores is not None else None
    risk       = score_obj.risk_label if score_obj else "unknown"

    reasoning = (
        f"Transaction {tx_id} has {n_illicit} illicit and {n_licit} licit "
        f"neighbours within {hops} hops. "
        + (f"GAE anomaly score={score_obj.gae_score:.3f}, "
           f"GraphSAGE fraud prob={score_obj.sage_prob:.3f}." if score_obj else "")
    )
    return ExplainResponse(
        tx_id                 = tx_id,
        risk_label            = risk,
        neighbourhood         = neighbourhood,
        n_illicit_neighbours  = n_illicit,
        n_licit_neighbours    = n_licit,
        risk_reasoning        = reasoning,
    )


@app.get("/drift/status", response_model=DriftStatusResponse)
def drift_status():
    """KS-test drift report on GAE anomaly score distribution."""
    if state.anomaly_scores is None or state.time_steps is None:
        raise HTTPException(503, "Models not loaded")
    report = detect_score_drift(state.anomaly_scores, state.time_steps)
    return DriftStatusResponse(**report.__dict__)


@app.get("/fraud-clusters")
def fraud_clusters():
    """Return fraud rings (connected illicit subgraphs) from Neo4j."""
    if not state.neo4j_ok:
        raise HTTPException(503, "Neo4j not connected")
    return {"clusters": state.neo4j_loader.get_fraud_clusters()}


@app.get("/bridge-nodes")
def bridge_nodes(threshold: float = 0.7):
    """Return licit nodes with high anomaly scores that connect to illicit nodes."""
    if not state.neo4j_ok:
        raise HTTPException(503, "Neo4j not connected")
    return {"bridge_nodes": state.neo4j_loader.get_bridge_nodes(threshold)}
