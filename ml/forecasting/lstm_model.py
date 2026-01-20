import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, input_size=4, hidden_size=32, num_layers=2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size)
        """
        lstm_out, _ = self.lstm(x)

        # Take output from last time step
        last_hidden = lstm_out[:, -1, :]

        out = self.fc(last_hidden)
        return out.squeeze(-1)
