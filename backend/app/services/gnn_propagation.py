"""
GNN Graph Propagation Service
Loads the pre-built GNN graph and propagates demand impacts through product relationships
"""
import torch
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Path to GNN graph files
GNN_MODEL_DIR = Path(__file__).parent.parent.parent.parent / "ml" / "models" / "gnn"
PRODUCT_NAMES_FILE = Path(__file__).parent.parent.parent.parent / "ml" / "data" / "raw" / "categories_products.csv"

# Add ML config directory to path for category relationships
ML_CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "ml" / "config"
if str(ML_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(ML_CONFIG_DIR))

try:
    from category_relationships import are_categories_related, get_propagation_multiplier, CATEGORY_NAMES
    CATEGORY_RULES_LOADED = True
except ImportError:
    CATEGORY_RULES_LOADED = False
    print("⚠️ Category relationship rules not found - using fallback propagation")

class GNNGraphPropagator:
    def __init__(self):
        self.adjacency_matrix = None
        self.sku_to_idx = None
        self.idx_to_sku = None
        self.metadata = None
        self.sku_to_name = {}  # SKU -> Product Name mapping
        self.graph_loaded = False
        self.use_category_rules = CATEGORY_RULES_LOADED
        self._load_graph()
        self._load_product_names()
    
    def _load_graph(self):
        """Load GNN graph files (adjacency matrix, SKU mappings, metadata)"""
        try:
            # Load adjacency matrix (PyTorch tensor - 240x240)
            adj_path = GNN_MODEL_DIR / "adjacency.pt"
            if adj_path.exists():
                # Use torch.load with weights_only=True for security
                self.adjacency_matrix = torch.load(adj_path, map_location='cpu', weights_only=True)
            
            # Load SKU to index mapping
            sku_to_idx_path = GNN_MODEL_DIR / "sku_to_idx.pt"
            if sku_to_idx_path.exists():
                self.sku_to_idx = torch.load(sku_to_idx_path, map_location='cpu', weights_only=True)
            
            # Load index to SKU mapping
            idx_to_sku_path = GNN_MODEL_DIR / "idx_to_sku.pt"
            if idx_to_sku_path.exists():
                self.idx_to_sku = torch.load(idx_to_sku_path, map_location='cpu', weights_only=True)
            
            # Load metadata (JSON - product list, categories, edge count)
            metadata_path = GNN_MODEL_DIR / "graph_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Validate all components loaded
            if all([
                self.adjacency_matrix is not None,
                self.sku_to_idx is not None,
                self.idx_to_sku is not None,
                self.metadata is not None
            ]):
                self.graph_loaded = True
                print(f"✅ GNN Graph loaded: {self.metadata['n_products']} products, {self.metadata['n_edges']} edges")
            else:
                print("⚠️ GNN Graph partially loaded - some files missing")
                
        except Exception as e:
            print(f"❌ Error loading GNN graph: {e}")
            self.graph_loaded = False
    
    def _load_product_names(self):
        """Load product name mapping from categories_products.csv"""
        try:
            import csv
            if PRODUCT_NAMES_FILE.exists():
                with open(PRODUCT_NAMES_FILE, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sku = row.get('SKU_ID', '').strip()
                        name = row.get('Product_Name', '').strip()
                        if sku and name:
                            self.sku_to_name[sku] = name
                print(f"✅ Loaded {len(self.sku_to_name)} product names")
            else:
                print(f"⚠️ Product names file not found: {PRODUCT_NAMES_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading product names: {e}")
    
    def get_product_name(self, sku: str) -> str:
        """Get product name for a SKU, fallback to SKU if not found"""
        return self.sku_to_name.get(sku, sku)
    
    def get_product_neighbors(self, sku: str, max_neighbors: int = 50) -> List[Tuple[str, float]]:
        """
        Get connected products (neighbors) from GNN graph with edge weights
        
        Args:
            sku: Product SKU (e.g., "SKU_GROC001")
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of (neighbor_sku, edge_weight) tuples sorted by weight descending
        """
        if not self.graph_loaded:
            return []
        
        try:
            # Get product index
            if sku not in self.sku_to_idx:
                return []
            
            prod_idx = self.sku_to_idx[sku]
            
            # Get row from adjacency matrix (all connections from this product)
            adjacency_row = self.adjacency_matrix[prod_idx]
            
            # Find non-zero edges (connected products)
            neighbors = []
            for neighbor_idx, edge_weight in enumerate(adjacency_row):
                if edge_weight > 0 and neighbor_idx != prod_idx:  # Exclude self-loops
                    neighbor_sku = self.idx_to_sku[neighbor_idx]
                    neighbors.append((neighbor_sku, float(edge_weight)))
            
            # Sort by edge weight descending (strongest relationships first)
            neighbors.sort(key=lambda x: x[1], reverse=True)
            
            return neighbors[:max_neighbors]
            
        except Exception as e:
            print(f"Error getting neighbors for {sku}: {e}")
            return []
    
    def propagate_impact(
        self, 
        affected_skus: List[str], 
        direct_multiplier: float,
        propagation_depth: int = 2,
        decay_factor: float = 0.5
    ) -> Dict[str, float]:
        """
        Propagate demand impact through GNN graph using edge weights AND category rules
        
        Args:
            affected_skus: List of directly affected product SKUs
            direct_multiplier: Demand multiplier for directly affected products (e.g., 1.15 for +15%)
            propagation_depth: How many hops to propagate through graph (1 = only neighbors, 2 = neighbors of neighbors)
            decay_factor: How much to reduce impact at each hop (0.5 = 50% reduction)
            
        Returns:
            Dict mapping SKU to final demand multiplier
        """
        if not self.graph_loaded:
            return {}
        
        # Initialize impact dict with directly affected products
        impacts = {}
        for sku in affected_skus:
            impacts[sku] = direct_multiplier
        
        # Propagate through graph using BFS-like approach
        current_layer = set(affected_skus)
        current_multiplier = direct_multiplier
        
        for depth in range(propagation_depth):
            next_layer = set()
            propagated_multiplier = 1.0 + (current_multiplier - 1.0) * decay_factor  # Decay the impact
            
            # For each product in current layer, propagate to neighbors
            for sku in current_layer:
                neighbors = self.get_product_neighbors(sku, max_neighbors=30)
                
                # Get source category for category-aware propagation
                source_category = self.get_category_for_sku(sku)
                
                for neighbor_sku, edge_weight in neighbors:
                    # Skip if already has stronger impact
                    if neighbor_sku in impacts:
                        continue
                    
                    # CATEGORY-AWARE FILTERING: Only propagate to related categories
                    if self.use_category_rules and source_category:
                        neighbor_category = self.get_category_for_sku(neighbor_sku)
                        
                        if neighbor_category and not are_categories_related(source_category, neighbor_category):
                            # Categories are not related - skip this neighbor
                            continue
                    
                    # Calculate propagated impact
                    if self.use_category_rules and source_category:
                        # Use category-aware multiplier
                        neighbor_category = self.get_category_for_sku(neighbor_sku)
                        if neighbor_category:
                            # Get category-based multiplier
                            category_multiplier = get_propagation_multiplier(
                                source_category, 
                                neighbor_category, 
                                propagated_multiplier
                            )
                            # Combine with edge weight
                            impact_delta = (category_multiplier - 1.0) * edge_weight
                            neighbor_impact = 1.0 + impact_delta
                        else:
                            # Fallback if category unknown
                            impact_delta = (propagated_multiplier - 1.0) * edge_weight * 0.3
                            neighbor_impact = 1.0 + impact_delta
                    else:
                        # Original logic: Propagated impact = propagated_multiplier * edge_weight
                        # Edge weight acts as correlation strength (0.0 to 1.0)
                        # Example: 1.15 direct, 0.5 decay, 0.8 edge_weight → 1.0 + (0.15 * 0.5 * 0.8) = 1.06
                        impact_delta = (propagated_multiplier - 1.0) * edge_weight
                        neighbor_impact = 1.0 + impact_delta
                    
                    impacts[neighbor_sku] = neighbor_impact
                    next_layer.add(neighbor_sku)
            
            # Move to next layer
            current_layer = next_layer
            current_multiplier = propagated_multiplier
            
            if not current_layer:
                break
        
        return impacts
    
    def get_category_for_sku(self, sku: str) -> Optional[str]:
        """Extract category from SKU name (e.g., SKU_GROC001 → GROC)"""
        if not sku.startswith("SKU_"):
            return None
        
        # Extract category part (between SKU_ and digits)
        # SKU_GROC001 → GROC, SKU_FTRW123 → FTRW
        import re
        match = re.match(r'SKU_([A-Z]+)\d+', sku)
        return match.group(1) if match else None
    
    def find_skus_by_category(self, category: str) -> List[str]:
        """Find all SKUs in a given category"""
        if not self.graph_loaded or not self.metadata:
            return []
        
        return [
            sku for sku in self.metadata['skus']
            if self.get_category_for_sku(sku) == category
        ]
    
    def find_skus_by_keyword(self, keywords: List[str]) -> List[str]:
        """
        Find SKUs matching keywords (category names or product patterns)
        
        Args:
            keywords: List of keywords like ['GROC', 'FRPR', 'rice', 'dairy']
            
        Returns:
            List of matching SKUs
        """
        if not self.graph_loaded or not self.metadata:
            return []
        
        matched_skus = []
        keywords_upper = [kw.upper() for kw in keywords]
        
        for sku in self.metadata['skus']:
            # Check if SKU category matches any keyword
            category = self.get_category_for_sku(sku)
            if category and category in keywords_upper:
                matched_skus.append(sku)
        
        return matched_skus


# Global singleton instance
_gnn_propagator = None

def get_gnn_propagator() -> GNNGraphPropagator:
    """Get global GNN propagator instance (singleton)"""
    global _gnn_propagator
    if _gnn_propagator is None:
        _gnn_propagator = GNNGraphPropagator()
    return _gnn_propagator
