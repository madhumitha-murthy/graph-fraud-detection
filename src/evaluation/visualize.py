"""
UMAP visualisation of fraud clusters in embedding space.

Generates:
    1. UMAP scatter: all labelled nodes coloured by fraud/licit label
    2. Anomaly score heatmap: node colour = GAE reconstruction error
    3. Temporal fraud rate chart: illicit % per time step
    4. Precision-Recall curve comparison: GAE vs GraphSAGE
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LOGS_DIR

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("umap-learn not installed — UMAP plots will be skipped.")


def plot_umap_fraud_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    anomaly_scores: np.ndarray | None = None,
    title: str = "Fraud Cluster Visualisation (UMAP)",
    save_path: Path | None = None,
) -> None:
    """
    2D UMAP projection of node embeddings coloured by fraud label.

    Args:
        embeddings:     (n_nodes, dim) — Node2Vec or GraphSAGE embeddings
        labels:         (n_nodes,) — 1=illicit, 0=licit, -1=unknown
        anomaly_scores: (n_nodes,) — optional GAE scores for heatmap
    """
    if not UMAP_AVAILABLE:
        return

    labelled = labels >= 0
    emb_lab  = embeddings[labelled]
    lbl_lab  = labels[labelled]

    print("Fitting UMAP (this may take 1-2 minutes)...")
    reducer  = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    proj     = reducer.fit_transform(emb_lab)

    n_plots = 2 if anomaly_scores is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    # ── Plot 1: Label colouring ──────────────────────────────────────────────
    ax = axes[0]
    colours = {0: "#2196F3", 1: "#F44336"}  # blue=licit, red=illicit
    labels_str = {0: "Licit", 1: "Illicit"}
    for cls in [0, 1]:
        m = lbl_lab == cls
        ax.scatter(
            proj[m, 0], proj[m, 1],
            c=colours[cls], label=labels_str[cls],
            alpha=0.4, s=4, rasterized=True,
        )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, framealpha=0.9)
    ax.grid(alpha=0.2)

    # ── Plot 2: Anomaly score heatmap ────────────────────────────────────────
    if anomaly_scores is not None:
        ax2      = axes[1]
        scores   = anomaly_scores[labelled]
        sc       = ax2.scatter(
            proj[:, 0], proj[:, 1],
            c=scores, cmap="RdYlGn_r",
            alpha=0.5, s=4, rasterized=True,
            norm=mcolors.Normalize(vmin=np.percentile(scores, 5),
                                   vmax=np.percentile(scores, 95)),
        )
        plt.colorbar(sc, ax=ax2, label="GAE Anomaly Score")
        ax2.set_title("Anomaly Score Heatmap (GAE)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("UMAP-1")
        ax2.set_ylabel("UMAP-2")
        ax2.grid(alpha=0.2)

    plt.tight_layout()
    out = save_path or (LOGS_DIR / "umap_fraud_clusters.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"UMAP plot saved: {out}")


def plot_temporal_fraud_rate(
    time_steps: np.ndarray,
    labels: np.ndarray,
    save_path: Path | None = None,
) -> None:
    """Fraud rate per time step — monitor for temporal drift."""
    rates = {}
    for t in sorted(np.unique(time_steps)):
        mask = (time_steps == t) & (labels >= 0)
        if mask.sum() == 0:
            continue
        rates[int(t)] = (labels[mask] == 1).mean()

    ts    = list(rates.keys())
    frate = list(rates.values())

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ts, frate, marker="o", markersize=4, linewidth=1.5, color="#F44336")
    ax.fill_between(ts, frate, alpha=0.15, color="#F44336")
    ax.axhline(np.mean(frate), linestyle="--", color="gray", linewidth=1, label=f"Mean={np.mean(frate):.2%}")
    ax.set_title("Temporal Fraud Rate by Time Step", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Fraud Rate")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = save_path or (LOGS_DIR / "temporal_fraud_rate.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Temporal fraud rate plot saved: {out}")


def plot_pr_curve(
    y_true: np.ndarray,
    gae_scores: np.ndarray,
    sage_scores: np.ndarray | None = None,
    save_path: Path | None = None,
) -> None:
    """Precision-Recall curve: GAE (unsupervised) vs GraphSAGE (supervised)."""
    from sklearn.metrics import precision_recall_curve, auc

    fig, ax = plt.subplots(figsize=(7, 5))

    prec, rec, _ = precision_recall_curve(y_true, gae_scores)
    ap = auc(rec, prec)
    ax.plot(rec, prec, label=f"GAE unsupervised (AP={ap:.3f})", linewidth=2)

    if sage_scores is not None:
        prec2, rec2, _ = precision_recall_curve(y_true, sage_scores)
        ap2 = auc(rec2, prec2)
        ax.plot(rec2, prec2, label=f"GraphSAGE supervised (AP={ap2:.3f})", linewidth=2)

    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="gray", label=f"Random baseline={baseline:.2%}")
    ax.set_title("Precision-Recall Curve — Fraud Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    out = save_path or (LOGS_DIR / "pr_curve.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"PR curve saved: {out}")
