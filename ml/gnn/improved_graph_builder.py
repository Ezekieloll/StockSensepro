"""
Improved GNN Graph Builder with Co-Purchase Analysis

This module builds a product relationship graph based on multiple signals:
1. Category-based connections (same category = related products)
2. Co-purchase patterns (products frequently bought together)
3. Temporal correlation (products with similar demand patterns)

The resulting graph is used by the GNN to capture product influences.
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import torch
import json


class ImprovedGraphBuilder:
    """
    Builds a weighted product graph using multiple relationship signals.
    """
    
    def __init__(self, transactions_path: str, categories_path: str):
        self.transactions_path = transactions_path
        self.categories_path = categories_path
        
        # Load data
        self.transactions = pd.read_csv(transactions_path)
        self.categories = pd.read_csv(categories_path)
        
        # Filter to sales only
        self.sales = self.transactions[self.transactions['event_type'] == 'sale'].copy()
        
        # Get unique SKUs
        self.all_skus = sorted(self.categories['SKU_ID'].unique())
        self.sku_to_idx = {sku: idx for idx, sku in enumerate(self.all_skus)}
        self.idx_to_sku = {idx: sku for sku, idx in self.sku_to_idx.items()}
        
        self.n_products = len(self.all_skus)
        print(f"✅ Loaded {len(self.sales):,} sales transactions")
        print(f"✅ {self.n_products} unique products")
    
    def _build_category_edges(self) -> np.ndarray:
        """
        Build edges between products in the same category.
        
        Logic: Products in the same category are substitutes/related.
        Example: SKU_BKDY001 (White Bread) <-> SKU_BKDY002 (Brown Bread)
        
        Returns:
            adj: Adjacency matrix (n_products x n_products)
        """
        print("\n📦 Building category-based edges...")
        
        adj = np.zeros((self.n_products, self.n_products))
        
        # Group SKUs by category
        category_skus = defaultdict(list)
        for _, row in self.categories.iterrows():
            category_skus[row['Category_Code']].append(row['SKU_ID'])
        
        # Connect all SKUs in the same category
        for category, skus in category_skus.items():
            for sku1, sku2 in combinations(skus, 2):
                if sku1 in self.sku_to_idx and sku2 in self.sku_to_idx:
                    i, j = self.sku_to_idx[sku1], self.sku_to_idx[sku2]
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0
        
        n_edges = int(adj.sum() / 2)
        print(f"   ✓ Created {n_edges} category edges")
        print(f"   ✓ Categories: {len(category_skus)}")
        
        return adj
    
    def _build_copurchase_edges(self, time_window_minutes: int = 30) -> np.ndarray:
        """
        Build edges based on products frequently bought together.
        
        Logic: Products bought by the same customer (same store, close in time)
        are likely complements.
        
        Example: Milk + Bread, Chips + Cola, Shampoo + Conditioner
        
        Args:
            time_window_minutes: Max time difference to consider same basket
            
        Returns:
            adj: Weighted adjacency matrix based on co-purchase frequency
        """
        print(f"\n🛒 Building co-purchase edges (window={time_window_minutes}min)...")
        
        # Convert timestamp to datetime
        self.sales['timestamp'] = pd.to_datetime(self.sales['timestamp'])
        
        # Sort by store and timestamp
        self.sales = self.sales.sort_values(['store_id', 'timestamp'])
        
        # Count co-purchases
        copurchase_count = defaultdict(int)
        
        # Process each store separately
        for store_id in self.sales['store_id'].unique():
            store_sales = self.sales[self.sales['store_id'] == store_id].copy()
            
            # Create time-based baskets
            basket_id = 0
            last_time = None
            basket_col = []
            
            for _, row in store_sales.iterrows():
                current_time = row['timestamp']
                
                if last_time is None or (current_time - last_time).total_seconds() > time_window_minutes * 60:
                    basket_id += 1
                
                basket_col.append(basket_id)
                last_time = current_time
            
            store_sales['basket_id'] = basket_col
            
            # Group by basket and count co-occurrences
            for _, basket in store_sales.groupby('basket_id'):
                products_in_basket = basket['product_id'].unique()
                
                if len(products_in_basket) >= 2:
                    for sku1, sku2 in combinations(products_in_basket, 2):
                        if sku1 in self.sku_to_idx and sku2 in self.sku_to_idx:
                            pair = tuple(sorted([sku1, sku2]))
                            copurchase_count[pair] += 1
        
        # Build adjacency matrix
        adj = np.zeros((self.n_products, self.n_products))
        
        for (sku1, sku2), count in copurchase_count.items():
            i, j = self.sku_to_idx[sku1], self.sku_to_idx[sku2]
            adj[i, j] = count
            adj[j, i] = count
        
        # Normalize to [0, 1]
        if adj.max() > 0:
            adj = adj / adj.max()
        
        n_edges = int((adj > 0).sum() / 2)
        print(f"   ✓ Found {n_edges} co-purchase pairs")
        print(f"   ✓ Max co-purchase count: {max(copurchase_count.values()) if copurchase_count else 0}")
        
        # Report top co-purchased pairs
        top_pairs = sorted(copurchase_count.items(), key=lambda x: -x[1])[:10]
        print("\n   📊 Top 10 Co-Purchased Product Pairs:")
        for (sku1, sku2), count in top_pairs:
            name1 = self._get_product_name(sku1)
            name2 = self._get_product_name(sku2)
            print(f"      {count:4d}x: {name1[:25]:<25} <-> {name2[:25]}")
        
        return adj
    
    def _build_temporal_correlation_edges(self, min_correlation: float = 0.5) -> np.ndarray:
        """
        Build edges based on temporal demand correlation.
        
        Logic: Products with similar daily demand patterns are related
        (e.g., both spike on weekends, both drop during holidays).
        
        Args:
            min_correlation: Minimum correlation to create an edge
            
        Returns:
            adj: Weighted adjacency matrix based on demand correlation
        """
        print(f"\n📈 Building temporal correlation edges (min_corr={min_correlation})...")
        
        # Aggregate daily demand per product
        daily_demand = self.sales.groupby(['date', 'product_id'])['quantity'].sum().unstack(fill_value=0)
        
        # Only keep products with enough data points
        min_days = 30
        valid_products = daily_demand.columns[daily_demand.sum() > min_days]
        daily_demand = daily_demand[valid_products]
        
        # Compute correlation matrix
        corr_matrix = daily_demand.corr()
        
        # Build adjacency matrix
        adj = np.zeros((self.n_products, self.n_products))
        
        for sku1 in corr_matrix.columns:
            for sku2 in corr_matrix.columns:
                if sku1 != sku2 and sku1 in self.sku_to_idx and sku2 in self.sku_to_idx:
                    corr = corr_matrix.loc[sku1, sku2]
                    if corr >= min_correlation:
                        i, j = self.sku_to_idx[sku1], self.sku_to_idx[sku2]
                        adj[i, j] = corr
                        adj[j, i] = corr
        
        n_edges = int((adj > 0).sum() / 2)
        print(f"   ✓ Found {n_edges} correlated product pairs")
        
        return adj
    
    def _get_product_name(self, sku: str) -> str:
        """Get product name from SKU."""
        match = self.categories[self.categories['SKU_ID'] == sku]
        if len(match) > 0:
            return match.iloc[0]['Product_Name']
        return sku
    
    def build_combined_graph(
        self, 
        category_weight: float = 0.3,
        copurchase_weight: float = 0.5,
        temporal_weight: float = 0.2,
        save_path: str = None
    ) -> np.ndarray:
        """
        Build combined graph from all signals.
        
        Args:
            category_weight: Weight for category edges
            copurchase_weight: Weight for co-purchase edges
            temporal_weight: Weight for temporal correlation edges
            save_path: Path to save the graph artifacts
            
        Returns:
            Combined weighted adjacency matrix
        """
        print("\n" + "="*60)
        print("🕸️  BUILDING IMPROVED PRODUCT GRAPH")
        print("="*60)
        
        # Build individual graphs
        category_adj = self._build_category_edges()
        copurchase_adj = self._build_copurchase_edges(time_window_minutes=30)
        temporal_adj = self._build_temporal_correlation_edges(min_correlation=0.5)
        
        # Combine with weights
        combined = (
            category_weight * category_adj +
            copurchase_weight * copurchase_adj +
            temporal_weight * temporal_adj
        )
        
        # Normalize to [0, 1]
        if combined.max() > 0:
            combined = combined / combined.max()
        
        # Apply threshold to remove weak edges
        threshold = 0.1
        combined[combined < threshold] = 0
        
        # Add self-loops
        np.fill_diagonal(combined, 1.0)
        
        n_edges = int((combined > 0).sum() - self.n_products) // 2
        density = n_edges / (self.n_products * (self.n_products - 1) / 2)
        
        print("\n" + "="*60)
        print("📊 GRAPH STATISTICS")
        print("="*60)
        print(f"   Nodes (products): {self.n_products}")
        print(f"   Edges: {n_edges}")
        print(f"   Density: {density:.4f}")
        print(f"   Avg edges per node: {n_edges * 2 / self.n_products:.1f}")
        
        # Save artifacts
        if save_path:
            self._save_graph(combined, save_path)
        
        return combined
    
    def _save_graph(self, adj: np.ndarray, save_path: str):
        """Save graph artifacts."""
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save adjacency matrix as tensor
        adj_tensor = torch.tensor(adj, dtype=torch.float32)
        torch.save(adj_tensor, path / 'adjacency.pt')
        
        # Save SKU mappings
        torch.save(self.sku_to_idx, path / 'sku_to_idx.pt')
        torch.save(self.idx_to_sku, path / 'idx_to_sku.pt')
        
        # Save metadata
        metadata = {
            'n_products': self.n_products,
            'n_edges': int((adj > 0).sum() - self.n_products) // 2,
            'categories': list(self.categories['Category_Code'].unique()),
            'skus': self.all_skus
        }
        with open(path / 'graph_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✅ Graph saved to: {save_path}")
        print(f"   - adjacency.pt")
        print(f"   - sku_to_idx.pt")
        print(f"   - idx_to_sku.pt")
        print(f"   - graph_metadata.json")
    
    def analyze_top_connections(self, adj: np.ndarray, top_n: int = 20):
        """Analyze and print top product connections."""
        print("\n" + "="*60)
        print(f"🔗 TOP {top_n} PRODUCT CONNECTIONS")
        print("="*60)
        
        # Get all edges with weights
        edges = []
        for i in range(self.n_products):
            for j in range(i + 1, self.n_products):
                if adj[i, j] > 0:
                    sku1 = self.idx_to_sku[i]
                    sku2 = self.idx_to_sku[j]
                    edges.append((sku1, sku2, adj[i, j]))
        
        # Sort by weight
        edges.sort(key=lambda x: -x[2])
        
        print(f"\n{'SKU 1':<15} {'Product 1':<25} {'SKU 2':<15} {'Product 2':<25} {'Weight':>8}")
        print("-" * 95)
        
        for sku1, sku2, weight in edges[:top_n]:
            name1 = self._get_product_name(sku1)[:25]
            name2 = self._get_product_name(sku2)[:25]
            print(f"{sku1:<15} {name1:<25} {sku2:<15} {name2:<25} {weight:>8.3f}")


def main():
    """Main function to build and analyze the improved graph."""
    
    # Paths
    transactions_path = "data/raw/transactions_3stores_2023_fullyear.csv"
    categories_path = "data/raw/categories_products.csv"
    output_path = "models/gnn"
    
    # Build graph
    builder = ImprovedGraphBuilder(transactions_path, categories_path)
    
    # UPDATED WEIGHTS (2026-02-06):
    # Increased category_weight to reduce spurious cross-category connections
    # Category propagation is now handled by category_relationships.py rules
    adj = builder.build_combined_graph(
        category_weight=0.7,      # ↑ Same category = meaningful relationship
        copurchase_weight=0.2,    # ↓ Contains noise (people buy unrelated items together)
        temporal_weight=0.1,      # ↓ Correlation ≠ causation
        save_path=output_path
    )
    
    # Analyze top connections
    builder.analyze_top_connections(adj, top_n=20)
    
    print("\n✅ Done! Graph is ready for GNN training.")
    
    return adj, builder.sku_to_idx


if __name__ == "__main__":
    adj, sku_to_idx = main()
