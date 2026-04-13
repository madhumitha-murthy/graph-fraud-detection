"""
Train Graph Autoencoder (GAE) for unsupervised fraud anomaly scoring.

Tracked in MLflow:
    - Training loss per epoch
    - Anomaly score distribution (mean, std, 95th percentile threshold)
    - AUC-ROC on labelled nodes using anomaly score as ranking signal
    - Model artefact (saved to models_saved/)
"""

import sys
from pathlib import Path

import numpy as np
import torch
import mlflow
import mlflow.pytorch
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    GAE_HIDDEN_DIM, GAE_LATENT_DIM, GAE_EPOCHS, GAE_LR,
    GAE_ANOMALY_PCTILE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT, MODELS_DIR,
)
from src.graph.builder import load_elliptic
from src.models.graph_autoencoder import GraphAutoencoder, gae_loss


def train(
    hidden_dim: int  = GAE_HIDDEN_DIM,
    latent_dim: int  = GAE_LATENT_DIM,
    epochs:     int  = GAE_EPOCHS,
    lr:         float = GAE_LR,
) -> tuple[GraphAutoencoder, np.ndarray, float]:
    """
    Train GAE and return (model, anomaly_scores, threshold).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data, meta = load_elliptic()
    data = data.to(device)

    model     = GraphAutoencoder(data.num_node_features, hidden_dim, latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="GAE_fraud_anomaly"):
        mlflow.log_params({
            "model":       "GraphAutoencoder",
            "hidden_dim":  hidden_dim,
            "latent_dim":  latent_dim,
            "epochs":      epochs,
            "lr":          lr,
            "n_nodes":     meta["n_nodes"],
            "n_edges":     meta["n_edges"],
            "illicit_pct": f"{meta['illicit_ratio']:.2%}",
        })

        # ── Training loop ────────────────────────────────────────────────────
        model.train()
        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            z   = model(data.x, data.edge_index)
            neg = negative_sampling(
                data.edge_index,
                num_nodes=data.num_nodes,
                num_neg_samples=data.edge_index.size(1),
            )
            loss = gae_loss(z, data.edge_index, neg, model)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if epoch % 20 == 0 or epoch == 1:
                print(f"  Epoch {epoch:4d}/{epochs}  loss={loss.item():.4f}")
                mlflow.log_metric("train_loss", loss.item(), step=epoch)

        # ── Anomaly scores ───────────────────────────────────────────────────
        print("\nComputing anomaly scores...")
        scores = model.anomaly_scores(data.x.cpu(), data.edge_index.cpu()).numpy()
        threshold = float(np.percentile(scores, GAE_ANOMALY_PCTILE))

        mlflow.log_metrics({
            "anomaly_score_mean":  float(scores.mean()),
            "anomaly_score_std":   float(scores.std()),
            "anomaly_threshold":   threshold,
        })

        # ── Evaluate on labelled nodes ───────────────────────────────────────
        labelled = data.y.cpu().numpy() >= 0
        y_true   = data.y.cpu().numpy()[labelled]
        y_score  = scores[labelled]

        auc_roc  = roc_auc_score(y_true, y_score)
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        auc_pr   = auc(rec, prec)

        print(f"\nGAE Anomaly Detection — AUC-ROC: {auc_roc:.4f}  AUC-PR: {auc_pr:.4f}")
        print(f"Threshold (p{GAE_ANOMALY_PCTILE}): {threshold:.4f}")

        mlflow.log_metrics({
            "auc_roc":     auc_roc,
            "auc_pr":      auc_pr,
            "threshold_p": GAE_ANOMALY_PCTILE,
        })

        # ── Save model ───────────────────────────────────────────────────────
        model_cpu = model.cpu()
        save_path = MODELS_DIR / "gae_model.pt"
        torch.save(model_cpu.state_dict(), save_path)
        mlflow.pytorch.log_model(model_cpu, "gae_model")
        np.save(MODELS_DIR / "gae_anomaly_scores.npy", scores)
        print(f"Model saved: {save_path}")

        mlflow.log_artifact(str(MODELS_DIR / "gae_anomaly_scores.npy"))

    return model_cpu, scores, threshold


if __name__ == "__main__":
    train()
