import torch
import torch.nn as nn


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, input_size)

        self.gate = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(input_size)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.elu(x)
        x = self.fc2(x)

        gate = self.sigmoid(self.gate(residual))
        x = gate * x + (1 - gate) * residual

        return self.layer_norm(x)


class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_vars, hidden_size):
        super().__init__()

        self.num_vars = num_vars

        self.weight_net = nn.Sequential(
            nn.Linear(num_vars, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_vars),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """
        x: (B, T, F, 1)
        returns: (B, T, F)
        """
        x = x.squeeze(-1)               # (B, T, F)

        # Compute feature importance weights
        weights = self.weight_net(x)    # (B, T, F)

        # Apply weights
        return x * weights

