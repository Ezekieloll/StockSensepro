"""
Demand Dataset V3 with GNN Support

This dataset extends V2 to also return:
1. Product index for GNN lookup
2. Supports batch-level GNN operations
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path

# Fixed weather encoding (consistent & explainable)
WEATHER_MAP = {
    "Cloudy": 0,
    "Rainy": 1,
    "Storm": 2,
    "Sunny": 3,
}


class DemandDatasetV3(Dataset):
    """
    Sliding-window dataset for demand forecasting with GNN support.
    
    Returns:
        x: Feature tensor (window_size, num_features)
        y: Target demand (scalar)
        sku_idx: Product index for GNN lookup
    """

    def __init__(
        self, 
        csv_path: str, 
        window_size: int = 30,
        sku_to_idx: dict = None
    ):
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size
        self.sku_to_idx = sku_to_idx or {}

        # Encode weather
        self.df["weather"] = self.df["weather"].map(WEATHER_MAP)

        # Sort for time consistency
        self.df = self.df.sort_values(
            ["store_id", "product_id", "date"]
        ).reset_index(drop=True)

        # Numeric feature columns (ORDER MATTERS)
        self.feature_cols = [
            "daily_demand",
            "price",
            "holiday_flag",
            "weather",
            "quantity_lag_7",
            "quantity_rolling_mean_7",
            "day_of_week",
            "month",
            "product_velocity",
        ]

        # Build index: group by SKU
        self.groups = []
        self.group_skus = []  # Store SKU ID for each group
        
        for (store_id, product_id), group in self.df.groupby(
            ["store_id", "product_id"]
        ):
            if len(group) > window_size:
                self.groups.append(group)
                self.group_skus.append(product_id)

        # Build sample index for fast __getitem__
        self.sample_index = []
        for group_idx, group in enumerate(self.groups):
            n_samples = len(group) - self.window_size
            for sample_idx in range(n_samples):
                self.sample_index.append((group_idx, sample_idx))

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        group_idx, sample_idx = self.sample_index[idx]
        group = self.groups[group_idx]
        sku_id = self.group_skus[group_idx]
        
        # Get window and target
        window = group.iloc[sample_idx : sample_idx + self.window_size]
        target_row = group.iloc[sample_idx + self.window_size]

        x = window[self.feature_cols].values
        y = target_row["daily_demand"]

        # Get SKU index for GNN
        sku_idx = self.sku_to_idx.get(sku_id, -1)

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        sku_idx = torch.tensor(sku_idx, dtype=torch.long)
        
        return x, y, sku_idx
    
    @property
    def num_features(self):
        return len(self.feature_cols)


def load_graph_data(graph_path: str = "models/gnn"):
    """
    Load the pre-built graph data.
    
    Returns:
        adj: Adjacency matrix tensor
        sku_to_idx: SKU to index mapping
        idx_to_sku: Index to SKU mapping
    """
    path = Path(graph_path)
    
    adj = torch.load(path / "adjacency.pt")
    sku_to_idx = torch.load(path / "sku_to_idx.pt")
    idx_to_sku = torch.load(path / "idx_to_sku.pt")
    
    return adj, sku_to_idx, idx_to_sku


def collate_with_sku(batch):
    """
    Custom collate function that handles SKU indices.
    """
    x_list, y_list, sku_list = zip(*batch)
    
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    sku_indices = torch.stack(sku_list)
    
    return x, y, sku_indices
