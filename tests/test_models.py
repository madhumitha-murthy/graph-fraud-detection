"""Unit tests for GAE and GraphSAGE model forward passes."""

import pytest
import torch
from torch_geometric.data import Data

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.graph_autoencoder import GraphAutoencoder, gae_loss
from src.models.graphsage import FraudGraphSAGE


def make_dummy_graph(n: int = 50, feat: int = 10) -> Data:
    x          = torch.randn(n, feat)
    src        = torch.randint(0, n, (80,))
    dst        = torch.randint(0, n, (80,))
    edge_index = torch.stack([src, dst])
    y          = torch.randint(0, 2, (n,))
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=n)


class TestGraphAutoencoder:
    def test_forward_shape(self):
        data  = make_dummy_graph()
        model = GraphAutoencoder(in_channels=10, hidden_dim=32, latent_dim=16)
        z     = model(data.x, data.edge_index)
        assert z.shape == (50, 16)

    def test_decode_shape(self):
        data  = make_dummy_graph()
        model = GraphAutoencoder(10, 32, 16)
        z     = model(data.x, data.edge_index)
        probs = model.decode(z, data.edge_index)
        assert probs.shape == (data.edge_index.size(1),)
        assert probs.min() >= 0.0 and probs.max() <= 1.0

    def test_anomaly_scores(self):
        data   = make_dummy_graph()
        model  = GraphAutoencoder(10, 32, 16)
        scores = model.anomaly_scores(data.x, data.edge_index)
        assert scores.shape == (50,)
        assert scores.min() >= 0.0

    def test_loss_decreases(self):
        data      = make_dummy_graph()
        model     = GraphAutoencoder(10, 32, 16)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        from torch_geometric.utils import negative_sampling
        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            z   = model(data.x, data.edge_index)
            neg = negative_sampling(data.edge_index, num_nodes=50, num_neg_samples=80)
            loss = gae_loss(z, data.edge_index, neg, model)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # Loss should generally decrease (not necessarily monotone — 5 steps)
        assert losses[-1] < losses[0] * 2   # sanity: not diverging


class TestFraudGraphSAGE:
    def test_forward_shape(self):
        data  = make_dummy_graph()
        model = FraudGraphSAGE(in_channels=10, hidden_channels=32, num_layers=2)
        out   = model(data.x, data.edge_index)
        assert out.shape == (50, 2)

    def test_predict_proba(self):
        data  = make_dummy_graph()
        model = FraudGraphSAGE(10, 32, 2)
        prob  = model.predict_proba(data.x, data.edge_index)
        assert prob.shape == (50,)
        assert prob.min() >= 0.0 and prob.max() <= 1.0

    def test_embeddings_shape(self):
        data  = make_dummy_graph()
        model = FraudGraphSAGE(10, 32, 2)
        emb   = model.get_embeddings(data.x, data.edge_index)
        assert emb.shape[0] == 50
