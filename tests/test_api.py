"""API endpoint tests using FastAPI TestClient."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.main import app, state
from src.api.schemas import FraudScore


@pytest.fixture(autouse=True)
def mock_state():
    """Inject minimal mock state so tests run without real data."""
    n = 100
    state.anomaly_scores = np.random.rand(n).astype(np.float32)
    state.threshold      = 0.7
    state.tx_to_idx      = {i: i for i in range(n)}
    state.time_steps     = np.random.randint(1, 50, size=n)
    state.labels         = np.random.choice([-1, 0, 1], size=n)
    state.n_nodes        = n
    state.gae_model      = None   # scores precomputed
    state.sage_model     = None
    state.neo4j_ok       = False
    state.data           = None
    yield


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["n_nodes_indexed"] == 100


def test_score_valid_tx():
    resp = client.post("/score", json={"tx_id": 0})
    assert resp.status_code == 200
    body = resp.json()
    assert "gae_score"      in body
    assert "sage_prob"      in body
    assert "ensemble_score" in body
    assert "risk_label"     in body
    assert body["risk_label"] in {"low", "medium", "high", "critical"}
    assert 0.0 <= body["ensemble_score"] <= 1.0


def test_score_unknown_tx():
    resp = client.post("/score", json={"tx_id": 99999})
    assert resp.status_code == 404


def test_batch_score():
    resp = client.post("/score/batch", json={"tx_ids": [0, 1, 2]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 3
    assert 0.0 <= body["flag_rate"] <= 1.0


def test_drift_status():
    resp = client.get("/drift/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "drifted"       in body
    assert "ks_statistic"  in body
    assert "recommendation" in body


def test_fraud_clusters_no_neo4j():
    resp = client.get("/fraud-clusters")
    assert resp.status_code == 503


def test_bridge_nodes_no_neo4j():
    resp = client.get("/bridge-nodes")
    assert resp.status_code == 503
