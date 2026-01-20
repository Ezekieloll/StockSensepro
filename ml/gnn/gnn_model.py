import torch
import torch.nn as nn


class SimpleGCN(nn.Module):
    """
    Simple Graph Convolution Network (GCN-style)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        """
        x   : (num_nodes, in_dim)
        adj : (num_nodes, num_nodes) normalized adjacency
        """
        x = torch.matmul(adj, x)   # message passing
        x = self.linear(x)
        return x


class GNNInfluence(nn.Module):
    """
    Applies GNN to enrich per-SKU embeddings
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.gcn = SimpleGCN(embed_dim, embed_dim)
        self.activation = nn.ReLU()

    def forward(self, sku_embeddings, adj):
        """
        sku_embeddings : (num_skus, embed_dim)
        adj            : (num_skus, num_skus)
        """
        out = self.gcn(sku_embeddings, adj)
        return self.activation(out)
