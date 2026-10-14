# GCN - learns spatial features between the stations
# shared LSTM - learns each node's temporal pattern

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.model_utils import PredictionHead, normalize_adjacency
from src.models.baselines.lstm import LSTMModel

class GraphConv(nn.Module):
    def __init__(self, in_dim, out_dim): # out (B, H, 2)
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x, a_hat):
        out = self.linear(x)
        out = F.relu(torch.einsum("nm,...md->...nd", a_hat, out))
        return out

class GCNLSTM(nn.Module):
    def __init__(self, in_dim, target_idx, horizon=1, adj=None,
                 hidden=64, gcn_layers=2, num_lstm_layers=1, dropout=0.5, **kwargs):
        super().__init__()
        assert adj is not None, "GCN-LSTM needs an adjacency matrix"
        self.target_idx = target_idx
        self.register_buffer("normalized_adj", normalize_adjacency(torch.as_tensor(adj, dtype=torch.float32)))
        # dims = [in_dim] + [hidden] * gcn_layers
        dims = [in_dim] + [in_dim] * gcn_layers
        lstm_in_dim = dims[-1]
        self.gcn = nn.ModuleList(GraphConv(dims[i], dims[i + 1]) for i in range(gcn_layers))
        self.drop = nn.Dropout(dropout)
        # self.lstm = nn.LSTM(lstm_in_dim, hidden, num_layers=num_lstm_layers, batch_first=True,
        #                     dropout=dropout if num_lstm_layers > 1 else 0.0)
        self.lstm = LSTMModel(lstm_in_dim, horizon, hidden, num_lstm_layers, dropout)
        self.norm = nn.LayerNorm(hidden)
        self.head = PredictionHead(hidden, horizon, dropout)

    def forward(self, x):
        B, T, N, _ = x.shape
        h = x
        for layer in self.gcn:
            h = self.drop(layer(h, self.normalized_adj))      # (B, T, N, F)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        out, _ = self.lstm(h)
        last = out[:, -1].reshape(B, N, -1)
        return self.head(self.norm(last[:, self.target_idx]))