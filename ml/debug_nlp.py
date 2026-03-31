"""Debug script to check if NLP similarity is being used in graph building."""

import sys
sys.path.insert(0, '.')

from gnn.improved_graph_builder import ImprovedGraphBuilder
import numpy as np

# Initialize builder
builder = ImprovedGraphBuilder(
    "data/raw/transactions_3stores_2023_fullyear.csv",
    "data/raw/categories_products.csv"
)

# Try to build NLP edges
print("\n🔍 Testing NLP similarity edge building...")
nlp_adj = builder._build_nlp_similarity_edges()

print(f"\n📊 NLP Adjacency Matrix Stats:")
print(f"   Shape: {nlp_adj.shape}")
print(f"   Non-zero edges: {(nlp_adj > 0).sum()}")
print(f"   Min (non-zero): {nlp_adj[nlp_adj > 0].min():.3f}")
print(f"   Max: {nlp_adj.max():.3f}")
print(f"   Unique values: {len(np.unique(nlp_adj))}")

# Check a specific product
sku_idx = builder.sku_to_idx.get('SKU_KICH001')
if sku_idx is not None:
    print(f"\n🍳 Frying Pan (SKU_KICH001) NLP edges:")
    connections = nlp_adj[sku_idx]
    non_zero_idx = np.where(connections > 0)[0]
    print(f"   Total connections: {len(non_zero_idx)}")
    if len(non_zero_idx) > 0:
        top_5 = np.argsort(connections)[::-1][:5]
        for i, idx in enumerate(top_5, 1):
            if connections[idx] > 0:
                other_sku = builder.idx_to_sku[idx]
                weight = connections[idx]
                print(f"   {i}. {other_sku} → {weight:.3f}")
