"""
Node2Vec graph embeddings — unsupervised structural representation learning.

Node2Vec (Grover & Leskovec, 2016) uses biased random walks to capture
both local (BFS) and community-level (DFS) graph structure:
    p < 1 → DFS → captures community / cluster membership → fraud rings
    q < 1 → DFS bias → good for detecting tightly-knit fraud clusters

These embeddings can be used:
    1. As features for downstream classifiers (no labels needed during training)
    2. For anomaly detection via clustering (KMeans + outlier scoring)
    3. For UMAP visualisation of fraud clusters
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.nn import Node2Vec as PyGNode2Vec
from torch_geometric.data import Data
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    N2V_DIM, N2V_WALK_LENGTH, N2V_CONTEXT_SIZE,
    N2V_WALKS_PER_NODE, N2V_P, N2V_Q,
    N2V_BATCH_SIZE, N2V_EPOCHS,
)


def train_node2vec(
    data: Data,
    device: torch.device,
    epochs: int = N2V_EPOCHS,
) -> tuple[PyGNode2Vec, np.ndarray]:
    """
    Train Node2Vec on the full transaction graph.

    Returns:
        model:      trained Node2Vec model
        embeddings: (n_nodes, dim) numpy array
    """
    model = PyGNode2Vec(
        data.edge_index,
        embedding_dim   = N2V_DIM,
        walk_length     = N2V_WALK_LENGTH,
        context_size    = N2V_CONTEXT_SIZE,
        walks_per_node  = N2V_WALKS_PER_NODE,
        p               = N2V_P,
        q               = N2V_Q,           # DFS bias → community detection
        num_negative_samples = 1,
        sparse          = True,
        num_nodes       = data.num_nodes,
    ).to(device)

    loader     = model.loader(batch_size=N2V_BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer  = torch.optim.SparseAdam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % max(1, epochs // 5) == 0:
            print(f"  Node2Vec epoch {epoch}/{epochs}  loss={total_loss/len(loader):.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(torch.arange(data.num_nodes, device=device)).cpu().numpy()

    return model, embeddings


def cluster_anomaly_scores(
    embeddings: np.ndarray,
    n_clusters: int = 20,
) -> np.ndarray:
    """
    Unsupervised anomaly scoring via KMeans clustering on Node2Vec embeddings.

    Score = distance from node to its nearest cluster centroid.
    Nodes far from all centroids are structurally isolated → suspicious.

    This requires NO labels — pure unsupervised fraud detection.
    """
    emb_norm = normalize(embeddings, norm="l2")
    km       = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(emb_norm)

    # Distance to nearest centroid
    centroids = km.cluster_centers_
    labels    = km.labels_
    dists     = np.linalg.norm(emb_norm - centroids[labels], axis=1)

    # Normalise to [0, 1]
    scores = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
    return scores.astype(np.float32)
