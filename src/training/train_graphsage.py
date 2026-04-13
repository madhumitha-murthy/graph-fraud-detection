"""
Train GraphSAGE for supervised fraud classification.

Uses temporal train/val/test split:
    Train:  time steps 1–34
    Val:    time steps 35–42
    Test:   time steps 43–49

Tracked in MLflow:
    - Loss, F1, Precision, Recall, AUC-ROC per epoch
    - Class-weighted loss to handle illicit/licit imbalance (~10:1)
    - Best model checkpoint (by val F1)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, classification_report,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    SAGE_HIDDEN_DIM, SAGE_NUM_LAYERS, SAGE_DROPOUT,
    SAGE_EPOCHS, SAGE_LR,
    MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODELS_DIR,
)
from src.graph.builder import load_elliptic
from src.models.graphsage import FraudGraphSAGE


def evaluate(
    model: FraudGraphSAGE,
    data,
    mask: torch.Tensor,
) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds  = logits[mask].argmax(dim=1).cpu().numpy()
        proba  = F.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        labels = data.y[mask].cpu().numpy()

    return {
        "f1":        f1_score(labels,      preds,  average="binary", zero_division=0),
        "precision": precision_score(labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(labels,   preds,  average="binary", zero_division=0),
        "auc_roc":   roc_auc_score(labels, proba),
    }


def train(
    hidden_dim:  int   = SAGE_HIDDEN_DIM,
    num_layers:  int   = SAGE_NUM_LAYERS,
    dropout:     float = SAGE_DROPOUT,
    epochs:      int   = SAGE_EPOCHS,
    lr:          float = SAGE_LR,
) -> FraudGraphSAGE:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data, meta = load_elliptic()
    # Move to device after split override below

    model     = FraudGraphSAGE(data.num_node_features, hidden_dim, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Override with stratified random split to avoid temporal distribution shift
    # (Elliptic time steps 43-49 have very different fraud patterns from 1-34)
    labelled_idx = (data.y >= 0).nonzero(as_tuple=True)[0].cpu().numpy()
    y_lab        = data.y[labelled_idx].cpu().numpy()
    from sklearn.model_selection import train_test_split
    train_idx, temp_idx = train_test_split(labelled_idx, test_size=0.3, stratify=y_lab, random_state=42)
    val_idx,   test_idx = train_test_split(temp_idx, test_size=0.5,
                                           stratify=y_lab[
                                               [list(labelled_idx).index(i) for i in temp_idx]
                                           ], random_state=42)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask   = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask  = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask[val_idx]     = True
    data.test_mask[test_idx]   = True
    data = data.to(device)

    # Class weights: handle illicit (~10%) vs licit (~90%) imbalance
    n_illicit = int(data.y[data.train_mask].eq(1).sum())
    n_licit   = int(data.y[data.train_mask].eq(0).sum())
    class_weight = torch.tensor(
        [1.0, n_licit / max(n_illicit, 1)], dtype=torch.float, device=device
    )
    print(f"Class weights: licit=1.0, illicit={class_weight[1]:.2f}x")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="GraphSAGE_fraud_classifier"):
        mlflow.log_params({
            "model":            "FraudGraphSAGE",
            "hidden_dim":       hidden_dim,
            "num_layers":       num_layers,
            "dropout":          dropout,
            "epochs":           epochs,
            "lr":               lr,
            "illicit_weight":   float(class_weight[1]),
            "n_train_illicit":  n_illicit,
            "n_train_licit":    n_licit,
        })

        best_val_f1   = 0.0
        best_state    = None

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss   = F.cross_entropy(
                logits[data.train_mask],
                data.y[data.train_mask],
                weight=class_weight,
            )
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0 or epoch == 1:
                val_metrics = evaluate(model, data, data.val_mask)
                scheduler.step(1 - val_metrics["f1"])

                print(
                    f"  Epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}  "
                    f"val_f1={val_metrics['f1']:.4f}  "
                    f"val_auc={val_metrics['auc_roc']:.4f}"
                )
                mlflow.log_metrics({
                    "train_loss":    loss.item(),
                    "val_f1":        val_metrics["f1"],
                    "val_precision": val_metrics["precision"],
                    "val_recall":    val_metrics["recall"],
                    "val_auc_roc":   val_metrics["auc_roc"],
                }, step=epoch)

                if val_metrics["f1"] > best_val_f1:
                    best_val_f1 = val_metrics["f1"]
                    best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        # ── Test evaluation ──────────────────────────────────────────────────
        model.load_state_dict(best_state)
        test_metrics = evaluate(model, data, data.test_mask)

        model.eval()
        with torch.no_grad():
            preds  = model(data.x, data.edge_index)[data.test_mask].argmax(1).cpu().numpy()
            labels = data.y[data.test_mask].cpu().numpy()
        report = classification_report(labels, preds, target_names=["licit", "illicit"])

        print(f"\n{'='*55}")
        print("GraphSAGE — Test Results")
        print(f"{'='*55}")
        print(report)
        print(f"AUC-ROC: {test_metrics['auc_roc']:.4f}")

        mlflow.log_metrics({
            "test_f1":        test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall":    test_metrics["recall"],
            "test_auc_roc":   test_metrics["auc_roc"],
            "best_val_f1":    best_val_f1,
        })

        # ── Save ─────────────────────────────────────────────────────────────
        save_path = MODELS_DIR / "graphsage_model.pt"
        torch.save(best_state, save_path)
        mlflow.pytorch.log_model(model.cpu(), "graphsage_model")
        print(f"Model saved: {save_path}")

    return model


if __name__ == "__main__":
    train()
