import torch.nn as nn

from src.models.model_utils import PredictionHead

class LSTMModel(nn.Module):
    def __init__(self, in_dim, horizon=1, hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = PredictionHead(hidden, horizon, dropout)

    def forward(self, x): # (B, T, F)
        out, _ = self.lstm(x)
        out = self.head(self.norm(out[:, -1]))
        return out
