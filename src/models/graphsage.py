"""
GraphSAGE — inductive GNN for supervised fraud classification.

GraphSAGE (Hamilton et al., 2017) learns by aggregating and transforming
neighbourhood features, enabling inductive inference on unseen nodes —
critical for real-world fraud detection where new transactions arrive constantly.

Architecture:
    3 × SAGEConv layers with BatchNorm + Dropout
    Final linear head → binary fraud probability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class FraudGraphSAGE(nn.Module):
    """
    Inductive GraphSAGE for node-level fraud classification.

    Why SAGEConv over GCNConv for production:
    - Inductive: works on new nodes at inference without full-graph retraining
    - Neighbourhood sampling: scales to 200k+ node graphs
    - Mean aggregation: robust to variable-degree nodes (common in fraud rings)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs   = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.dropout = dropout

        # Input → hidden
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden → hidden
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Hidden → half hidden
        self.convs.append(SAGEConv(hidden_channels, hidden_channels // 2))
        self.bns.append(nn.BatchNorm1d(hidden_channels // 2))

        # Classification head
        self.classifier = nn.Linear(hidden_channels // 2, 2)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return penultimate-layer embeddings (for UMAP visualisation)."""
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return fraud probability (softmax over logits)."""
        logits = self.forward(x, edge_index)
        return F.softmax(logits, dim=1)[:, 1]   # P(fraud)
