# GCN - learns spatial features between the stations
# shared LSTM - learns each node's temporal pattern
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_utils import PredictionHead, normalize_adjacency
from src.models.baselines.lstm import LSTMModel
from src.utils.paths import EXTERNAL_DIR


def get_adjacency():
    adj_matrix_path = os.path.join(EXTERNAL_DIR, "adjacency_matrix.npy")
    adj = np.load(adj_matrix_path) if Path(str(adj_matrix_path)).exists() else None

    return adj

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim): # out (B, H, 2)
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, a_hat):
        out = self.linear(x)
        out = F.relu(torch.einsum("nm,...md->...nd", a_hat, out))
        return out

class GCNLSTM(nn.Module):
    def __init__(self, in_dim, horizon=1,
                 hidden=64, gcn_layers=2, lstm_layers=1, dropout=0.5, **kwargs):
        super().__init__()
        adj = get_adjacency()
        assert adj is not None, "GCN-LSTM needs an adjacency matrix"
        self.horizon = horizon
        self.register_buffer("normalized_adj", normalize_adjacency(torch.as_tensor(adj, dtype=torch.float32)))
        dims = [in_dim] + [hidden] * gcn_layers
        # dims = [in_dim] + [in_dim] * gcn_layers
        lstm_in_dim = dims[-1]
        self.gcn = nn.ModuleList(GraphConv(dims[i], dims[i + 1]) for i in range(gcn_layers))
        self.drop = nn.Dropout(dropout)
        self.lstm = LSTMModel(lstm_in_dim, self.horizon, hidden, lstm_layers, dropout)

    def forward(self, x):
        B, T, N, _ = x.shape
        h = x
        for layer in self.gcn:
            h = self.drop(layer(h, self.normalized_adj))      # (B, T, N, F)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1) # (B*N, T, F)
        out = self.lstm(h)
        last = out.reshape(B, N, self.horizon, -1)
        last = last.permute(0, 2, 1, 3)  # (B, H, N, 2)

        return last