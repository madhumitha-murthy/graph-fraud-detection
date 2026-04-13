"""Central configuration — all tunable parameters in one place."""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
DATA_DIR   = ROOT_DIR / "data" / "elliptic"
MODELS_DIR = ROOT_DIR / "models_saved"
LOGS_DIR   = ROOT_DIR / "logs"

for _d in (DATA_DIR, MODELS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Elliptic dataset filenames ───────────────────────────────────────────────
FEATURES_FILE  = DATA_DIR / "elliptic_txs_features.csv"
EDGELIST_FILE  = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_FILE   = DATA_DIR / "elliptic_txs_classes.csv"

# ── Graph Autoencoder ────────────────────────────────────────────────────────
GAE_HIDDEN_DIM    = 128
GAE_LATENT_DIM    = 64
GAE_EPOCHS        = 200
GAE_LR            = 1e-3
GAE_ANOMALY_PCTILE = 95        # reconstruction error percentile → fraud threshold

# ── GraphSAGE ────────────────────────────────────────────────────────────────
SAGE_HIDDEN_DIM   = 256
SAGE_NUM_LAYERS   = 3
SAGE_DROPOUT      = 0.3
SAGE_EPOCHS       = 150
SAGE_LR           = 5e-4
SAGE_BATCH_SIZE   = 1024

# ── Node2Vec ─────────────────────────────────────────────────────────────────
N2V_DIM           = 128
N2V_WALK_LENGTH   = 20
N2V_CONTEXT_SIZE  = 10
N2V_WALKS_PER_NODE = 10
N2V_P             = 1.0        # return parameter
N2V_Q             = 0.5        # in-out parameter (< 1 = DFS → community structure)
N2V_BATCH_SIZE    = 256
N2V_EPOCHS        = 5

# ── Drift detection ──────────────────────────────────────────────────────────
DRIFT_WINDOW      = 5          # time steps
DRIFT_KS_ALPHA    = 0.05
DRIFT_ZSCORE_THRESH = 3.0

# ── Neo4j ────────────────────────────────────────────────────────────────────
NEO4J_URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# ── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", str(ROOT_DIR / "mlruns"))
MLFLOW_EXPERIMENT    = "graph-fraud-detection"

# ── Labels (Elliptic convention) ─────────────────────────────────────────────
LABEL_ILLICIT = 1
LABEL_LICIT   = 2
LABEL_UNKNOWN = "unknown"
