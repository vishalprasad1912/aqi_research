import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.baselines.lstm import LSTMModel
from src.models.baselines.gcn_lstm import get_adjacency

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.5):
        super().__init__()
        assert out_dim % heads == 0
        self.h, self.dim_h = heads, out_dim // heads
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.dim_h))     # score of the neighbour j
        self.a_dst = nn.Parameter(torch.empty(heads, self.dim_h))     # score of the receiving node i
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj):
        z = self.W(x).view(*x.shape[:-1], self.h, self.dim_h) # (B,T,N,h,dh)
        s_src = (z * self.a_src).sum(-1).transpose(-1, -2)          # (B,T,h,N)
        s_dst = (z * self.a_dst).sum(-1).transpose(-1, -2)          # (B,T,h,N)
        e = F.leaky_relu(s_dst.unsqueeze(-1) + s_src.unsqueeze(-2), 0.2)   # (B,T,h,N_i,N_j)
        e = e.masked_fill(adj == 0, -1e9)                           # only connected stations
        a = self.drop(F.softmax(e, dim=-1))                         # attention weights per node
        out = torch.einsum("...hij,...jhd->...ihd", a, z)           # weighted sum of neighbors
        return F.elu(out.reshape(*x.shape[:-1], -1))                # (B,T,N,out)

class GATLSTM(nn.Module):
    def __init__(self, in_dim, horizon=1,
                 hidden=64, heads=4, gat_layers=2, lstm_layers=1, dropout=0.5, **kwargs):
        super().__init__()
        adj = get_adjacency()
        assert adj is not None, "GAT-LSTM needs an adjacency matrix"
        self.register_buffer("adj", torch.as_tensor(adj, dtype=torch.float32))
        dims = [in_dim] + [hidden] * gat_layers
        # dims = [in_dim] + [in_dim] * gat_layers

        self.gat = nn.ModuleList(GATLayer(dims[i], dims[i + 1], heads, dropout)
                                 for i in range(gat_layers))
        self.drop = nn.Dropout(dropout)
        self.lstm = LSTMModel(dims[-1], self.horizon, hidden, lstm_layers, dropout)

    def forward(self, x):
        B, T, N, _ = x.shape
        h = x
        for layer in self.gat:
            h = self.drop(layer(h, self.adj)) # (B,T,N,F)
        h = h.permute(0, 2, 1, 3).reshape(B * N, T, -1) # (B*N, T, F)
        out = self.lstm(h)
        last = out.reshape(B, N, self.horizon, -1)
        last = last.permute(0, 2, 1, 3)  # (B, H, N, 2)

        return last