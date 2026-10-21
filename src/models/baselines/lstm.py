import torch.nn as nn

from src.models.model_utils import PredictionHead

class LSTMModel(nn.Module):
    def __init__(self, in_dim, horizon=1, hidden=64, lstm_layers=1, dropout=0.5):
        super().__init__()

        self.lstms = nn.ModuleList()
        prev_dim = in_dim
        for layer in range(lstm_layers):
            self.lstms.append(
                nn.LSTM(prev_dim, hidden * (lstm_layers - layer), batch_first=True,
                    dropout=dropout if layer > 1 else 0
                )
            )
            prev_dim = hidden * (lstm_layers - layer)

        self.norm = nn.LayerNorm(hidden)
        self.head = PredictionHead(hidden, horizon, dropout)

    def forward(self, x): # (B, T, F)
        for lstm in self.lstms:
            x, _ = lstm(x) # out is (B, T, H)
        return self.head(self.norm(x[:, -1])) # (B, H)
