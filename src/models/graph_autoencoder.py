"""
Graph Autoencoder (GAE) for unsupervised fraud anomaly scoring.

Architecture:
    Encoder: 2-layer GCN → latent embedding Z
    Decoder: inner-product  A_hat = sigmoid(Z · Z^T)
    Loss:    binary cross-entropy on edge reconstruction

Anomaly score = per-node reconstruction error (how poorly the graph
can reconstruct a node's neighbourhood). High score → structurally
anomalous → potential fraud node.

Reference: Kipf & Welling, "Variational Graph Auto-Encoders" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import (
    negative_sampling,
    add_self_loops,
    to_dense_adj,
)


class GCNEncoder(nn.Module):
    """Two-layer GCN encoder producing node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels,  hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.bn1   = nn.BatchNorm1d(hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=0.3, training=self.training)
        return self.conv2(x, edge_index)


class GraphAutoencoder(nn.Module):
    """
    GAE with inner-product decoder.

    forward() returns the latent embeddings Z.
    Use decode() for link reconstruction probability.
    Use anomaly_scores() for per-node fraud scoring.
    """

    def __init__(self, in_channels: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = GCNEncoder(in_channels, hidden_dim, latent_dim)

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def decode(
        self,
        z: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Inner-product decoder: P(edge | z_i, z_j) = sigmoid(z_i · z_j)."""
        src, dst = edge_index
        return torch.sigmoid((z[src] * z[dst]).sum(dim=1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encode(x, edge_index)

    @torch.no_grad()
    def anomaly_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int = 4096,
    ) -> torch.Tensor:
        """
        Per-node anomaly score = mean reconstruction error over the node's edges.

        For each node i:
            score(i) = mean over neighbours j of (1 - P(edge_ij | Z))
                     + mean over non-neighbours k of P(edge_ik | Z)

        High score → node's neighbourhood is poorly reconstructed → structural anomaly.
        """
        self.eval()
        z   = self.encode(x, edge_index)
        src, dst = edge_index

        # Positive edge reconstruction error
        pos_scores = torch.zeros(x.size(0), device=x.device)
        pos_counts = torch.zeros(x.size(0), device=x.device)
        for i in range(0, src.size(0), batch_size):
            s = src[i : i + batch_size]
            d = dst[i : i + batch_size]
            p = torch.sigmoid((z[s] * z[d]).sum(dim=1))
            err = 1.0 - p
            pos_scores.scatter_add_(0, s, err)
            pos_counts.scatter_add_(0, s, torch.ones_like(err))

        pos_counts = pos_counts.clamp(min=1)
        return (pos_scores / pos_counts).cpu()


def gae_loss(
    z: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    model: GraphAutoencoder,
) -> torch.Tensor:
    """Binary cross-entropy loss on positive and negative edges."""
    pos_pred = model.decode(z, pos_edge_index)
    neg_pred = model.decode(z, neg_edge_index)

    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
    return pos_loss + neg_loss
