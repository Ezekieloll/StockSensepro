import torch
import torch.nn as nn
from forecasting.tft_layers import VariableSelectionNetwork


class TemporalFusionTransformer(nn.Module):
    def __init__(
        self,
        num_features,
        hidden_size=128,
        lstm_layers=2,
        attention_heads=4,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Variable Selection
        self.vsn = VariableSelectionNetwork(
            num_vars=num_features,
            hidden_size=hidden_size,
        )


        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Temporal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        x = x.unsqueeze(-1)               # (B, T, F, 1)
        x = self.vsn(x)                   # (B, T, F)

        lstm_out, _ = self.lstm(x)        # (B, T, H)

        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        last_hidden = attn_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)
