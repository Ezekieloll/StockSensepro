import torch
import torch.nn as nn

from forecasting.lstm_model import LSTMBaseline
from gnn.gnn_model import GNNInfluence


class LSTMGNNModel(nn.Module):
    """
    LSTM + GNN hybrid model for demand forecasting
    """

    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()

        # Temporal encoder (per SKU)
        self.lstm = LSTMBaseline(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
        )

        # GNN influence module (across SKUs)
        self.gnn = GNNInfluence(embed_dim=hidden_size)

        # Final prediction head
        self.fc_out = nn.Linear(hidden_size, 1)

    def forward(self, x, sku_indices, adj):
        """
        x           : (batch, seq_len, input_size)
        sku_indices : (batch,) → index of SKU in graph
        adj         : (num_skus, num_skus)
        """

        # LSTM forward
        lstm_out, _ = self.lstm.lstm(x)
        sku_embed = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Prepare full SKU embedding matrix
        num_skus = adj.size(0)
        device = x.device

        sku_embeddings = torch.zeros(
            (num_skus, sku_embed.size(1)), device=device
        )

        # Scatter batch embeddings into full SKU table
        sku_embeddings[sku_indices] = sku_embed

        # GNN propagation
        gnn_out = self.gnn(sku_embeddings, adj)

        # Gather back batch embeddings
        enriched = gnn_out[sku_indices]

        # Final prediction
        out = self.fc_out(enriched)

        return out.squeeze(-1)
