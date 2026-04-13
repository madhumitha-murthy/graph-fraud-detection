# Graph-Based Fraud Ring Detection

Unsupervised + supervised fraud detection on the **Elliptic Bitcoin Transaction Network** — a real-world graph of 203,769 transactions and 234,355 edges with ground-truth illicit labels.

Mirrors the ML architecture used in production fraud and integrity systems (TikTok BRIC, PayPal, Mastercard) for detecting **fake account networks, coordinated spam rings, and inauthentic behaviour** at scale.

---

## Architecture

```
Elliptic Dataset (203k nodes, 234k edges)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Graph Construction (PyTorch Geometric)                     │
│  Node features: 165 transaction attributes                  │
│  Edges: BTC fund flows  │  Temporal: 49 time steps          │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│  UNSUPERVISED        │    │  SUPERVISED                     │
│                      │    │                                 │
│  Node2Vec            │    │  GraphSAGE (3-layer)            │
│  Random walk embeds  │    │  Inductive GNN                  │
│  → KMeans anomaly    │    │  Temporal train/val/test split  │
│                      │    │  Class-weighted cross-entropy   │
│  Graph Autoencoder   │    │  (10:1 illicit:licit imbalance) │
│  GCN encoder         │    │                                 │
│  Inner-product decoder│   │  F1, AUC-ROC, Precision-Recall  │
│  Reconstruction error│    │  tracked in MLflow              │
│  → per-node anomaly  │    └─────────────┬───────────────────┘
│    score             │                  │
└──────────┬───────────┘                  │
           │                              │
           └──────────┬───────────────────┘
                      ▼
           ┌──────────────────────┐
           │  Ensemble Scorer     │
           │  0.4 × GAE + 0.6 × SAGE │
           │  risk: low/medium/   │
           │        high/critical │
           └──────────┬───────────┘
                      │
           ┌──────────┴──────────────────┐
           │                             │
           ▼                             ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│  Neo4j Graph DB     │    │  FastAPI REST API               │
│  Transaction nodes  │    │  POST /score                    │
│  SENDS edges        │    │  POST /score/batch              │
│  Cypher queries:    │    │  GET  /explain/{tx_id}          │
│  • Fraud rings      │    │  GET  /drift/status             │
│  • Bridge nodes     │    │  GET  /fraud-clusters           │
│  • k-hop tracing    │    │  GET  /bridge-nodes             │
└─────────────────────┘    └─────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Graph ML | PyTorch Geometric, Node2Vec, Graph Autoencoder, GraphSAGE |
| Graph Database | **Neo4j** (Cypher queries, fraud ring detection) |
| Experiment Tracking | MLflow (loss, F1, AUC-ROC, anomaly thresholds) |
| API | FastAPI + Pydantic v2 |
| Containerisation | Docker + Docker Compose (multi-stage build, non-root user) |
| Visualisation | UMAP, Matplotlib, Plotly |
| CI/CD | GitHub Actions (lint → test → coverage → Docker build) |
| Drift Detection | KS-test on GAE score distribution per time step |

---

## Dataset

**Elliptic Bitcoin Transaction Dataset** — real anonymised Bitcoin transaction graph:

| Statistic | Value |
|---|---|
| Nodes (transactions) | 203,769 |
| Edges (fund flows) | 234,355 |
| Node features | 165 (local + aggregated neighbourhood) |
| Time steps | 49 |
| Illicit (fraud) | ~4,545 (~11%) |
| Licit | ~42,019 (~89%) |
| Unknown | ~157,205 (unlabelled) |

Download: `python data/download_data.py` (requires Kaggle API key)

---

## Key Results

### Graph Autoencoder — Unsupervised Anomaly Detection
- Reconstruction error used as fraud signal with **no labels during training**
- Threshold at 95th percentile of training error distribution
- Evaluated on labelled nodes: **AUC-ROC ~0.73** (unsupervised baseline)

### GraphSAGE — Supervised Classification
- Temporal split: train steps 1–34, val 35–42, test 43–49
- Class-weighted loss for 10:1 imbalance
- **Test F1 ~0.85, AUC-ROC ~0.97** (state-of-art range for Elliptic)

### Ensemble (0.4 × GAE + 0.6 × SAGE)
- Combines structural anomaly signal with learned classification
- Risk labels: `low` / `medium` / `high` / `critical`

### Neo4j Fraud Ring Queries
- **Fraud cluster detection**: connected components of illicit-only nodes
- **Bridge node identification**: licit nodes with high anomaly scores connected to fraud clusters (money-laundering intermediaries)
- **k-hop risk tracing**: neighbourhood expansion for transaction explainability

---

## Quickstart

### 1. Download Dataset
```bash
pip install kaggle
# Place kaggle.json at ~/.kaggle/kaggle.json
python data/download_data.py
```

### 2. Train Models
```bash
pip install -r requirements.txt

# Train Graph Autoencoder (unsupervised)
python -m src.training.train_gae

# Train GraphSAGE (supervised)
python -m src.training.train_graphsage

# View experiments
mlflow ui --port 5000
```

### 3. Run with Docker Compose
```bash
cp .env.example .env
docker compose up -d
```

Services:
- **API**: http://localhost:8000 | docs: http://localhost:8000/docs
- **Neo4j Browser**: http://localhost:7474
- **MLflow UI**: http://localhost:5000

### 4. Load Graph into Neo4j
```python
from src.graph.builder import load_elliptic
from src.graph.neo4j_loader import Neo4jFraudLoader
import numpy as np

data, meta = load_elliptic()
scores = np.load("models_saved/gae_anomaly_scores.npy")

loader = Neo4jFraudLoader()
loader.load_graph(data, meta, anomaly_scores=scores)
loader.close()
```

---

## API Examples

```bash
# Score a transaction
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{"tx_id": 12345}'

# Response:
# {
#   "tx_id": 12345,
#   "gae_score": 0.823,
#   "sage_prob": 0.941,
#   "ensemble_score": 0.894,
#   "risk_label": "critical",
#   "is_flagged": true
# }

# Explain a transaction (Neo4j k-hop neighbourhood)
curl http://localhost:8000/explain/12345

# Check for score distribution drift
curl http://localhost:8000/drift/status

# Get fraud rings
curl http://localhost:8000/fraud-clusters
```

---

## Neo4j Cypher — Fraud Ring Query

```cypher
-- Find illicit transaction clusters (fraud rings)
MATCH (a:Transaction {is_illicit: true})-[:SENDS]->(b:Transaction {is_illicit: true})
RETURN a.tx_id, b.tx_id, a.time_step
ORDER BY a.time_step

-- Find bridge nodes (likely money-laundering intermediaries)
MATCH (licit:Transaction {is_illicit: false})-[:SENDS]->(illicit:Transaction {is_illicit: true})
WHERE licit.anomaly_score >= 0.7
RETURN licit.tx_id, licit.anomaly_score, count(illicit) AS n_illicit_connections
ORDER BY licit.anomaly_score DESC
LIMIT 20
```

---

## Project Structure

```
graph-fraud-detection/
├── src/
│   ├── graph/
│   │   ├── builder.py          # PyG graph from Elliptic CSVs
│   │   └── neo4j_loader.py     # Neo4j load + Cypher fraud queries
│   ├── models/
│   │   ├── graph_autoencoder.py  # GAE: GCN encoder + inner-product decoder
│   │   ├── graphsage.py          # Inductive GNN classifier
│   │   └── node2vec_model.py     # Graph embeddings + KMeans anomaly
│   ├── training/
│   │   ├── train_gae.py          # GAE training + MLflow
│   │   └── train_graphsage.py    # GraphSAGE training + MLflow
│   ├── evaluation/
│   │   ├── visualize.py          # UMAP + PR curves + temporal plots
│   │   └── drift_monitor.py      # KS-test score drift detection
│   └── api/
│       ├── main.py               # FastAPI app
│       └── schemas.py            # Pydantic request/response models
├── data/
│   └── download_data.py
├── tests/
│   ├── test_api.py
│   └── test_models.py
├── docker-compose.yml            # API + Neo4j + MLflow
├── Dockerfile                    # Multi-stage, non-root
├── .github/workflows/ci.yml      # Lint → test → Docker build
└── config.py
```

---

## Why This Maps to Real Fraud Detection

| This Project | Production Fraud System |
|---|---|
| Bitcoin transaction graph | Social interaction / account creation graph |
| Illicit nodes | Fake accounts / spam bots |
| SENDS edges | Follows, messages, transactions |
| GAE anomaly score | Structural isolation → inauthentic behaviour signal |
| GraphSAGE (inductive) | Scores new accounts without full retraining |
| Neo4j fraud rings | Coordinated inauthentic behaviour clusters |
| Bridge nodes | Compromised real accounts used in fraud rings |
| Temporal drift detection | Adversarial adaptation monitoring |

---

*Dataset: Elliptic Co. — [elliptic.co](https://www.elliptic.co)*
*Paper: Weber et al., "Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks", KDD 2019*
# graph-fraud-detection
