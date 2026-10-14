import torch.nn as nn

from src.models.model_utils import PredictionHead

class LSTMModel(nn.Module):
    def __init__(self, in_dim, horizon=1, hidden=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden)
        self.head = PredictionHead(hidden, horizon, dropout)

    def forward(self, x):
        seq = x                                      # (B, T, F)
        out, _ = self.lstm(seq)                      # (B, T, H)
        return self.head(self.norm(out[:, -1]))      # (B, H)