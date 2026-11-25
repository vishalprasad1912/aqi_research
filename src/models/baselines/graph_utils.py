from pathlib import Path

import numpy as np
import torch

from src.utils.paths import EXTERNAL_DIR


def load_graph(weighted=True):
    filename = ("weighted_adjacency_matrix.npy" if weighted else "adjacency_matrix.npy")
    path = Path(EXTERNAL_DIR) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing adjacency matrix: {path}")
    adj = np.load(path).astype(np.float32)
    # Convert nonzero matrix entries into graph edges.
    target, source = np.nonzero(adj)
    edge_index = torch.tensor(
        np.stack([source, target]),
        dtype=torch.long
    )
    edge_weight = torch.tensor(
        adj[target, source],
        dtype=torch.float32
    )
    return edge_index, edge_weight, adj.shape[0]

def batch_edges(edge_index, edge_weight, batch_size, num_nodes):
    """
    Create disconnected copies of the same graph.
    batch_size here represents B*T.
    """
    offsets = (torch.arange(batch_size, device=edge_index.device) * num_nodes)
    batched_edge_index = (
        edge_index.unsqueeze(0)
        + offsets[:, None, None]
    )
    batched_edge_index = (
        batched_edge_index.permute(1, 0, 2)
        .reshape(2, -1)
    )
    batched_edge_weight = edge_weight.repeat(batch_size)
    return batched_edge_index, batched_edge_weight
