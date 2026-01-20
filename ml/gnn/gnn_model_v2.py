"""
Improved GNN Model with Weighted Edge Support

This module provides an improved GNN that:
1. Uses weighted adjacency matrices (not just 0/1)
2. Has proper normalization for message passing
3. Supports multi-layer GCN with residual connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedGCNLayer(nn.Module):
    """
    Graph Convolution Layer that properly handles weighted edges.
    
    Message Passing: h' = σ(D^(-1/2) A D^(-1/2) H W)
    
    Where:
    - A = weighted adjacency matrix
    - D = degree matrix (sum of edge weights per node)
    - H = node features
    - W = learnable weights
    """
    
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)
    
    @staticmethod
    def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """
        Symmetric normalization: D^(-1/2) A D^(-1/2)
        This ensures proper message passing with weighted edges.
        """
        # Compute degree (sum of edge weights for each node)
        degree = adj.sum(dim=1)
        
        # Avoid division by zero
        degree = torch.clamp(degree, min=1e-6)
        
        # D^(-1/2)
        d_inv_sqrt = degree.pow(-0.5)
        
        # D^(-1/2) A D^(-1/2)
        d_mat = torch.diag(d_inv_sqrt)
        normalized = d_mat @ adj @ d_mat
        
        return normalized
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with weighted message passing.
        
        Args:
            x: Node features (num_nodes, in_dim)
            adj: Weighted adjacency matrix (num_nodes, num_nodes)
            
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        # Normalize adjacency (do this once, not every forward)
        adj_norm = self.normalize_adjacency(adj)
        
        # Message passing: aggregate neighbor features weighted by edge weights
        aggregated = torch.matmul(adj_norm, x)
        
        # Transform
        out = self.linear(aggregated)
        
        return out


class MultiLayerGNN(nn.Module):
    """
    Multi-layer GNN with residual connections and dropout.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        hidden_dim: int = None,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or embed_dim
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # First layer
        self.layers.append(WeightedGCNLayer(embed_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(WeightedGCNLayer(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Output layer (if more than 1 layer)
        if num_layers > 1:
            self.layers.append(WeightedGCNLayer(hidden_dim, embed_dim))
            self.norms.append(nn.LayerNorm(embed_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        # Residual projection if dimensions differ
        self.residual_proj = None
        if hidden_dim != embed_dim and num_layers > 1:
            self.residual_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all GNN layers.
        """
        residual = x
        
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, adj)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Residual connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        
        return x + residual


class GNNInfluenceV2(nn.Module):
    """
    Improved GNN module for enriching SKU embeddings with graph information.
    
    This module:
    1. Takes per-SKU temporal embeddings from LSTM/TFT
    2. Propagates information through the product graph
    3. Returns enriched embeddings that capture product relationships
    """
    
    def __init__(
        self, 
        embed_dim: int,
        num_gnn_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gnn = MultiLayerGNN(
            embed_dim=embed_dim,
            hidden_dim=embed_dim,
            num_layers=num_gnn_layers,
            dropout=dropout
        )
        
        # Gate for combining original and GNN-enriched embeddings
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
    
    def forward(
        self, 
        sku_embeddings: torch.Tensor, 
        adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Enrich SKU embeddings using graph structure.
        
        Args:
            sku_embeddings: Per-SKU embeddings (num_skus, embed_dim)
            adj: Weighted adjacency matrix (num_skus, num_skus)
            
        Returns:
            Enriched embeddings (num_skus, embed_dim)
        """
        # GNN forward pass
        gnn_out = self.gnn(sku_embeddings, adj)
        
        # Gated combination: learns how much to use GNN vs original
        gate_input = torch.cat([sku_embeddings, gnn_out], dim=-1)
        gate = self.gate(gate_input)
        
        # Weighted combination
        enriched = gate * gnn_out + (1 - gate) * sku_embeddings
        
        return enriched
