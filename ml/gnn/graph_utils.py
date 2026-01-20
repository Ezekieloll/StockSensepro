import torch
import pandas as pd
from gnn.graph_builder import build_category_graph


def build_graph_tensors(csv_path):
    nodes, edges = build_category_graph(csv_path)

    node_to_idx = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for u, v in edges:
        i, j = node_to_idx[u], node_to_idx[v]
        adj[i, j] = 1.0

    # Add self-loops
    adj += torch.eye(num_nodes)

    # Normalize adjacency (D^-1 A)
    deg = adj.sum(dim=1, keepdim=True)
    adj = adj / deg

    return nodes, node_to_idx, adj


if __name__ == "__main__":
    nodes, node_to_idx, adj = build_graph_tensors(
        "data/raw/categories_products.csv"
    )

    print("Adjacency shape:", adj.shape)
    print("First 5 nodes:", nodes[:5])
