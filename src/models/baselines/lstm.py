import torch.nn as nn

from src.models.model_utils import PredictionHead

class LSTMModel(nn.Module):
    def __init__(self, in_dim, horizon=1, hidden=64, lstm_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, horizon * 2)
        )
        self.horizon = horizon

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.head(out)
        return out.reshape(-1, self.horizon, 2)