"""
TFT + GNN Hybrid Model for Demand Forecasting

This model combines:
1. Temporal Fusion Transformer (TFT) for time-series patterns
2. Graph Neural Network (GNN) for product relationship learning

The GNN captures:
- Substitute effects (similar products in same category)
- Complement effects (products bought together)
- Demand correlation patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from gnn.gnn_model_v2 import GNNInfluenceV2


class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder for time-series features.
    """
    
    def __init__(
        self, 
        num_features: int, 
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_size)
        
        # LSTM for temporal encoding
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, seq_len, num_features)
            
        Returns:
            Temporal encoding (batch, hidden_size)
        """
        # Project input
        x = self.input_proj(x)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        temporal_embed = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Normalize
        temporal_embed = self.layer_norm(temporal_embed)
        
        return temporal_embed


class VariableSelectionNetwork(nn.Module):
    """
    Learns which features are important for each sample.
    """
    
    def __init__(self, num_features: int, hidden_size: int):
        super().__init__()
        
        # Feature-wise processing
        self.grn = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_features),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, seq_len, num_features)
            
        Returns:
            Weighted features (batch, seq_len, num_features)
        """
        # Average across time dimension for weight calculation
        x_mean = x.mean(dim=1)  # (batch, num_features)
        
        # Compute feature weights
        weights = self.grn(x_mean)  # (batch, num_features)
        
        # Apply weights
        x_weighted = x * weights.unsqueeze(1)
        
        return x_weighted


class TFTWithGNN(nn.Module):
    """
    Temporal Fusion Transformer + Graph Neural Network
    
    Architecture:
    1. Variable Selection: Learn important features
    2. Temporal Encoder: LSTM for time patterns
    3. GNN: Enrich embeddings with product relationships
    4. Output Head: Final demand prediction
    """
    
    def __init__(
        self,
        num_features: int = 9,
        hidden_size: int = 128,
        lstm_layers: int = 2,
        num_gnn_layers: int = 2,
        attention_heads: int = 4,
        dropout: float = 0.1,
        num_products: int = 240
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_products = num_products
        
        # 1. Variable Selection
        self.var_selection = VariableSelectionNetwork(num_features, hidden_size)
        
        # 2. Temporal Encoder
        self.temporal_encoder = TemporalEncoder(
            num_features=num_features,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=dropout
        )
        
        # 3. Multi-head Self-Attention (optional TFT component)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # 4. GNN for product relationships
        self.gnn = GNNInfluenceV2(
            embed_dim=hidden_size,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # 5. Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # Initialize product embedding buffer (will be filled during forward)
        self.register_buffer(
            'product_embeddings',
            torch.zeros(num_products, hidden_size)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        sku_indices: torch.Tensor = None,
        adj: torch.Tensor = None,
        use_gnn: bool = True
    ) -> torch.Tensor:
        """
        Forward pass through TFT+GNN.
        
        Args:
            x: Input features (batch, seq_len, num_features)
            sku_indices: Product indices for GNN lookup (batch,)
            adj: Weighted adjacency matrix (num_products, num_products)
            use_gnn: Whether to use GNN enrichment
            
        Returns:
            predictions: Demand predictions (batch,)
        """
        batch_size = x.size(0)
        device = x.device
        
        # 1. Variable Selection
        x = self.var_selection(x)
        
        # 2. Temporal Encoding
        temporal_embed = self.temporal_encoder(x)  # (batch, hidden_size)
        
        # 3. GNN Enrichment (if enabled and graph available)
        if use_gnn and sku_indices is not None and adj is not None:
            # Build full product embedding matrix
            # Initialize with zeros
            product_embeds = torch.zeros(
                self.num_products, self.hidden_size, 
                device=device
            )
            
            # Filter valid indices (not -1)
            valid_mask = sku_indices >= 0
            valid_indices = sku_indices[valid_mask]
            valid_embeds = temporal_embed[valid_mask]
            
            if len(valid_indices) > 0:
                # Scatter batch embeddings into product matrix
                product_embeds.index_copy_(0, valid_indices, valid_embeds)
                
                # GNN propagation
                enriched_products = self.gnn(product_embeds, adj)
                
                # Gather back the enriched embeddings for batch
                enriched_embed = enriched_products[valid_indices]
                
                # Replace valid embeddings
                temporal_embed = temporal_embed.clone()
                temporal_embed[valid_mask] = enriched_embed
        
        # 4. Output prediction
        out = self.output_head(temporal_embed)
        
        return out.squeeze(-1)


class TFTWithGNNLight(nn.Module):
    """
    Lighter version of TFT+GNN for faster training.
    
    Simplified architecture:
    1. Input projection
    2. LSTM encoder
    3. Optional GNN enrichment
    4. Output head
    """
    
    def __init__(
        self,
        num_features: int = 9,
        hidden_size: int = 64,
        lstm_layers: int = 2,
        dropout: float = 0.1,
        num_products: int = 240
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_products = num_products
        
        # Input projection
        self.input_proj = nn.Linear(num_features, hidden_size)
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Simple GNN (single layer for speed)
        self.gnn_linear = nn.Linear(hidden_size, hidden_size)
        self.gnn_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
        # Output head
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(
        self, 
        x: torch.Tensor, 
        sku_indices: torch.Tensor = None,
        adj: torch.Tensor = None,
        use_gnn: bool = True
    ) -> torch.Tensor:
        """Forward pass."""
        device = x.device
        batch_size = x.size(0)
        
        # Input projection
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        embed = lstm_out[:, -1, :]  # Last timestep
        embed = self.layer_norm(embed)
        
        # GNN enrichment
        if use_gnn and sku_indices is not None and adj is not None:
            # Build product embedding matrix
            product_embeds = torch.zeros(
                self.num_products, self.hidden_size, device=device
            )
            
            valid_mask = sku_indices >= 0
            valid_indices = sku_indices[valid_mask]
            
            if len(valid_indices) > 0:
                product_embeds.index_copy_(0, valid_indices, embed[valid_mask])
                
                # Normalize adjacency
                deg = adj.sum(dim=1, keepdim=True).clamp(min=1e-6)
                adj_norm = adj / deg
                
                # Message passing
                aggregated = torch.matmul(adj_norm, product_embeds)
                transformed = self.gnn_linear(aggregated)
                transformed = F.gelu(transformed)
                
                # Gated combination
                enriched = transformed[valid_indices]
                original = embed[valid_mask]
                gate_input = torch.cat([original, enriched], dim=-1)
                gate = self.gnn_gate(gate_input)
                combined = gate * enriched + (1 - gate) * original
                
                embed = embed.clone()
                embed[valid_mask] = combined
        
        # Output
        out = self.output(embed)
        return out.squeeze(-1)
