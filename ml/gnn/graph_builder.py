import pandas as pd
from itertools import combinations
import torch

def build_category_graph(csv_path):
    """
    Builds an undirected product graph based on category membership.

    Returns:
        nodes: sorted list of product_ids
        edges: list of (src, dst) tuples
    """
    df = pd.read_csv(csv_path)

    # Normalize column names (safety)
    df = df.rename(
        columns={
            "SKU_ID": "product_id",
            "Category_Code": "category",
        }
    )

    nodes = sorted(df["product_id"].unique().tolist())
    edges = set()

    # Group products by category
    for _, group in df.groupby("category"):
        products = group["product_id"].tolist()

        # Fully connect products within the same category
        for u, v in combinations(products, 2):
            edges.add((u, v))
            edges.add((v, u))  # undirected

    return nodes, list(edges)


if __name__ == "__main__":
    nodes, edges = build_category_graph("data/raw/categories_products.csv")

    print(f"Number of products (nodes): {len(nodes)}")
    print(f"Number of edges: {len(edges)}")

    # Build index mapping
    sku_to_idx = {sku: i for i, sku in enumerate(nodes)}

    # Build adjacency matrix
    num_nodes = len(nodes)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for u, v in edges:
        i, j = sku_to_idx[u], sku_to_idx[v]
        adj[i, j] = 1.0

    # Save artifacts
    torch.save(adj, "data/processed2/adjacency.pt")
    torch.save(sku_to_idx, "data/processed2/sku_to_idx.pt")

    print("✅ Saved adjacency.pt and sku_to_idx.pt")
