import pandas as pd
import torch
from torch.utils.data import Dataset

# Fixed weather encoding (consistent & explainable)
WEATHER_MAP = {
    "Cloudy": 0,
    "Rainy": 1,
    "Storm": 2,
    "Sunny": 3,
}


class DemandDatasetV2(Dataset):
    """
    Sliding-window dataset for demand forecasting (v2 features).
    """

    def __init__(self, csv_path, window_size=17):
        self.df = pd.read_csv(csv_path)
        self.window_size = window_size

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
        for (_, _), group in self.df.groupby(
            ["store_id", "product_id"]
        ):
            if len(group) > window_size:
                self.groups.append(group)

    def __len__(self):
        return sum(len(g) - self.window_size for g in self.groups)

    def __getitem__(self, idx):
        for group in self.groups:
            if idx < len(group) - self.window_size:
                window = group.iloc[idx : idx + self.window_size]
                target_row = group.iloc[idx + self.window_size]

                x = window[self.feature_cols].values
                y = target_row["daily_demand"]
                product_id = target_row["product_id"]

                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y, dtype=torch.float32) 
                return x, y

            idx -= len(group) - self.window_size

        raise IndexError
