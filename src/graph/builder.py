"""
Build a PyTorch Geometric graph from the Elliptic Bitcoin dataset.

The Elliptic dataset contains 203,769 Bitcoin transactions (nodes) across
49 time steps, with 234,355 directed edges. Each node has 167 features:
  - Col 0   : transaction ID
  - Col 1   : time step (1–49)
  - Cols 2–94  : local transaction features
  - Cols 95–166: aggregated neighbourhood features

Labels: 1 = illicit (fraud), 2 = licit, "unknown" = unlabelled.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FEATURES_FILE, EDGELIST_FILE, CLASSES_FILE, LABEL_ILLICIT, LABEL_LICIT


def load_elliptic(
    normalise: bool = True,
    include_unknown: bool = False,
) -> tuple[Data, dict]:
    """
    Load the Elliptic dataset into a PyG Data object.

    Args:
        normalise:        StandardScale node features.
        include_unknown:  If False, mask unknown-label nodes from loss.

    Returns:
        data: PyG Data with fields:
              x, edge_index, y, train_mask, val_mask, test_mask,
              time_step, tx_id, illicit_ratio
        meta: dict with label mappings and dataset stats
    """
    # ── Load raw CSVs ────────────────────────────────────────────────────────
    feat_df  = pd.read_csv(FEATURES_FILE, header=None)
    edge_df  = pd.read_csv(EDGELIST_FILE)
    class_df = pd.read_csv(CLASSES_FILE)

    # Standardise column names
    feat_df.columns  = ["txid", "time_step"] + [f"f{i}" for i in range(feat_df.shape[1] - 2)]
    class_df.columns = ["txid", "class"]

    # ── Build node index mapping ─────────────────────────────────────────────
    all_txids = feat_df["txid"].values
    tx_to_idx = {tx: i for i, tx in enumerate(all_txids)}
    n_nodes   = len(all_txids)

    # ── Node features ────────────────────────────────────────────────────────
    feature_cols = [c for c in feat_df.columns if c.startswith("f")]
    X = feat_df[feature_cols].values.astype(np.float32)

    if normalise:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)

    # ── Edge index ───────────────────────────────────────────────────────────
    src_col, dst_col = edge_df.columns[0], edge_df.columns[1]
    valid_edges = edge_df[
        edge_df[src_col].isin(tx_to_idx) & edge_df[dst_col].isin(tx_to_idx)
    ]
    src = np.array([tx_to_idx[t] for t in valid_edges[src_col].values])
    dst = np.array([tx_to_idx[t] for t in valid_edges[dst_col].values])
    # Make undirected (Bitcoin tx graph — message passing in both directions)
    edge_index = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])], axis=0
    )

    # ── Labels ───────────────────────────────────────────────────────────────
    label_map  = {"1": LABEL_ILLICIT, "2": LABEL_LICIT, 1: LABEL_ILLICIT, 2: LABEL_LICIT}
    class_df["label"] = class_df["class"].map(
        lambda x: label_map.get(str(x).strip(), -1)
    )
    class_dict = dict(zip(class_df["txid"], class_df["label"]))

    y = np.array([class_dict.get(tx, -1) for tx in all_txids], dtype=np.int64)
    # Remap: illicit=1→1, licit=2→0, unknown=-1
    binary_y = np.where(y == LABEL_ILLICIT, 1, np.where(y == LABEL_LICIT, 0, -1))

    # ── Temporal split (time steps 1–34 train, 35–42 val, 43–49 test) ───────
    time_step = feat_df["time_step"].values
    labelled  = binary_y >= 0

    train_mask = labelled & (time_step <= 34)
    val_mask   = labelled & (time_step >= 35) & (time_step <= 42)
    test_mask  = labelled & (time_step >= 43)

    # ── Assemble PyG Data ────────────────────────────────────────────────────
    data = Data(
        x          = torch.tensor(X,           dtype=torch.float),
        edge_index = torch.tensor(edge_index,  dtype=torch.long),
        y          = torch.tensor(binary_y,    dtype=torch.long),
        train_mask = torch.tensor(train_mask,  dtype=torch.bool),
        val_mask   = torch.tensor(val_mask,    dtype=torch.bool),
        test_mask  = torch.tensor(test_mask,   dtype=torch.bool),
        time_step  = torch.tensor(time_step,   dtype=torch.long),
        tx_id      = torch.tensor(all_txids,   dtype=torch.long),
    )

    n_illicit = int((binary_y == 1).sum())
    n_licit   = int((binary_y == 0).sum())

    meta = {
        "n_nodes":       n_nodes,
        "n_edges":       edge_index.shape[1] // 2,
        "n_features":    X.shape[1],
        "n_illicit":     n_illicit,
        "n_licit":       n_licit,
        "n_unknown":     int((binary_y == -1).sum()),
        "illicit_ratio": n_illicit / max(n_illicit + n_licit, 1),
        "n_time_steps":  int(time_step.max()),
        "tx_to_idx":     tx_to_idx,
    }

    print(
        f"Graph loaded: {n_nodes:,} nodes | {meta['n_edges']:,} edges | "
        f"{X.shape[1]} features | "
        f"illicit={n_illicit:,} ({meta['illicit_ratio']:.1%}) | "
        f"licit={n_licit:,} | unknown={meta['n_unknown']:,}"
    )
    return data, meta


def get_timestep_subgraph(data: Data, time_step: int) -> Data:
    """Extract the subgraph for a single time step (for temporal analysis)."""
    mask = data.time_step == time_step
    node_idx = mask.nonzero(as_tuple=True)[0]
    node_set  = set(node_idx.tolist())

    # Filter edges where both endpoints are in this time step
    src, dst = data.edge_index
    edge_mask = torch.tensor(
        [s.item() in node_set and d.item() in node_set
         for s, d in zip(src, dst)],
        dtype=torch.bool,
    )

    return Data(
        x          = data.x[node_idx],
        edge_index = data.edge_index[:, edge_mask],
        y          = data.y[node_idx],
        time_step  = data.time_step[node_idx],
        tx_id      = data.tx_id[node_idx],
    )
