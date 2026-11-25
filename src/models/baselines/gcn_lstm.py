from src.models.baselines.graph_utils import load_graph, batch_edges
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLSTM(nn.Module):
    def __init__(self, in_dim, horizon=1, hidden=64, gcn_layers=2, lstm_layers=1, target_idx=None, dropout=0.5, **kwargs):
        super().__init__()
        edge_index, edge_weight, num_nodes = load_graph(weighted=True)
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.target_idx = target_idx
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        # Spatial learning
        self.gcn = nn.ModuleList()
        dims = [in_dim] + [hidden] * gcn_layers
        for i in range(gcn_layers):
            self.gcn.append(GCNConv(dims[i], dims[i + 1], cached=False, normalize=True))
        self.dropout = nn.Dropout(dropout)
        # Shared temporal learning
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden)
        # Multi-step, multi-feature prediction
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, 2 * horizon)
        )

    def forward(self, x):
        B, T, N, F_in = x.shape
        h = x.reshape(B * T * N, F_in) # [B,T,N,F] -> [B*T*N,F]
        # Duplicate graph for each batch/timestep
        edge_index, edge_weight = batch_edges(self.edge_index, self.edge_weight, B * T, N)
        # Spatial graph convolution
        for layer in self.gcn:
            h = layer(h, edge_index, edge_weight)
            h = F.relu(h)
            h = self.dropout(h)
        h = h.reshape(B, T, N, -1) # [B*T*N,hidden] -> [B,T,N,hidden]
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1) # [B,T,N,hidden] -> [B*N,T,hidden]
        # Shared temporal LSTM
        out, _ = self.lstm(h)
        # Last temporal representation
        out = self.norm(out[:, -1, :])
        # Predict H steps for each station
        out = self.head(out)
        out = out.reshape(B, N, self.horizon, 2)
        out = out.permute(0, 2, 1, 3) # [B,N,H,2] -> [B,H,N,2]
        out = out[:, :, self.target_idx, :] # [B,H,2]
        return out