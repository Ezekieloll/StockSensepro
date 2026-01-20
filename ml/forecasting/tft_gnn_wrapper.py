"""
TFT + GNN Wrapper Model

This wraps the existing tested TemporalFusionTransformer
and adds GNN enrichment on top.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from forecasting.tft_model import TemporalFusionTransformer


class TFTWithGNNWrapper(nn.Module):
    """
    Wraps the existing TFT model and adds GNN enrichment.
    
    Architecture:
    1. Use existing TFT for temporal encoding
    2. Add GNN layer to enrich predictions with product relationships
    """
    
    def __init__(
        self,
        num_features: int = 9,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        num_products: int = 240,
        use_gnn: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_products = num_products
        self.use_gnn = use_gnn
        
        # Use the ORIGINAL tested TFT model
        self.tft = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            attention_heads=attention_heads
        )
        
        # GNN components (only if enabled)
        if use_gnn:
            # GNN linear layer
            self.gnn_linear = nn.Linear(hidden_size, hidden_size)
            
            # Gating mechanism
            self.gnn_gate = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Sigmoid()
            )
            
            # Output head (replaces TFT's fc layer)
            self.output = nn.Linear(hidden_size, 1)
            
            # Layer norm
            self.layer_norm = nn.LayerNorm(hidden_size)
    
    def _get_tft_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the hidden state from TFT before the final fc layer.
        We need to replicate the TFT forward pass up to the last hidden state.
        """
        # Replicate TFT forward pass
        x = x.unsqueeze(-1)               # (B, T, F, 1)
        x = self.tft.vsn(x)               # (B, T, F)
        
        lstm_out, _ = self.tft.lstm(x)    # (B, T, H)
        
        attn_out, _ = self.tft.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        last_hidden = attn_out[:, -1, :]  # (B, H)
        return last_hidden
    
    def forward(
        self, 
        x: torch.Tensor, 
        sku_indices: torch.Tensor = None,
        adj: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with optional GNN enrichment.
        """
        # If GNN not enabled or no graph provided, use original TFT
        if not self.use_gnn or sku_indices is None or adj is None:
            return self.tft(x)
        
        device = x.device
        batch_size = x.size(0)
        
        # Get TFT hidden state
        hidden = self._get_tft_hidden(x)  # (batch, hidden_size)
        
        # Build product embedding matrix
        product_embeds = torch.zeros(
            self.num_products, self.hidden_size, device=device
        )
        
        # Filter valid indices (not -1)
        valid_mask = sku_indices >= 0
        valid_indices = sku_indices[valid_mask]
        
        if len(valid_indices) > 0:
            # Scatter batch embeddings into product matrix
            product_embeds.index_copy_(0, valid_indices, hidden[valid_mask])
            
            # Normalize adjacency for message passing
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
            adj_norm = adj / degree
            
            # GNN message passing
            aggregated = torch.matmul(adj_norm, product_embeds)
            transformed = self.gnn_linear(aggregated)
            transformed = F.gelu(transformed)
            transformed = self.layer_norm(transformed)
            
            # Gather enriched embeddings for batch
            enriched = transformed[valid_indices]
            original = hidden[valid_mask]
            
            # Gated combination
            gate_input = torch.cat([original, enriched], dim=-1)
            gate = self.gnn_gate(gate_input)
            combined = gate * enriched + (1 - gate) * original
            
            # Update hidden with GNN-enriched version
            hidden = hidden.clone()
            hidden[valid_mask] = combined
        
        # Final prediction
        out = self.output(hidden)
        return out.squeeze(-1)


class TFTEnsemble(nn.Module):
    """
    Ensemble of TFT (temporal) and GNN (relational) predictions.
    
    Final prediction = α × TFT_pred + (1-α) × GNN_enriched_pred
    Where α is learned.
    """
    
    def __init__(
        self,
        num_features: int = 9,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        attention_heads: int = 4,
        num_products: int = 240
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_products = num_products
        
        # Original TFT model
        self.tft = TemporalFusionTransformer(
            num_features=num_features,
            hidden_size=hidden_size,
            lstm_layers=lstm_layers,
            attention_heads=attention_heads
        )
        
        # Learnable ensemble weight
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # GNN prediction branch
        self.gnn_linear = nn.Linear(hidden_size, hidden_size)
        self.gnn_output = nn.Linear(hidden_size, 1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def _get_tft_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Get TFT hidden state."""
        x = x.unsqueeze(-1)
        x = self.tft.vsn(x)
        lstm_out, _ = self.tft.lstm(x)
        attn_out, _ = self.tft.attention(lstm_out, lstm_out, lstm_out)
        return attn_out[:, -1, :]
    
    def forward(
        self, 
        x: torch.Tensor, 
        sku_indices: torch.Tensor = None,
        adj: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass with ensemble prediction."""
        
        # TFT prediction (always)
        tft_pred = self.tft(x)
        
        # If no graph, return TFT only
        if sku_indices is None or adj is None:
            return tft_pred
        
        device = x.device
        hidden = self._get_tft_hidden(x)
        
        # Build and propagate through GNN
        product_embeds = torch.zeros(
            self.num_products, self.hidden_size, device=device
        )
        
        valid_mask = sku_indices >= 0
        valid_indices = sku_indices[valid_mask]
        
        gnn_pred = torch.zeros_like(tft_pred)
        
        if len(valid_indices) > 0:
            product_embeds.index_copy_(0, valid_indices, hidden[valid_mask])
            
            # GNN propagation
            degree = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
            adj_norm = adj / degree
            
            aggregated = torch.matmul(adj_norm, product_embeds)
            transformed = self.gnn_linear(aggregated)
            transformed = F.gelu(transformed)
            transformed = self.layer_norm(transformed)
            
            # GNN prediction
            gnn_out = self.gnn_output(transformed[valid_indices]).squeeze(-1)
            gnn_pred[valid_mask] = gnn_out
        
        # Ensemble: learned weighted average
        alpha = torch.sigmoid(self.alpha)  # Ensure 0-1 range
        final_pred = alpha * tft_pred + (1 - alpha) * gnn_pred
        
        return final_pred
