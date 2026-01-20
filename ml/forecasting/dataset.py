import pandas as pd
import torch
from torch.utils.data import Dataset


class DemandDataset(Dataset):
    def __init__(self, csv_path, window_size=17):
        self.window_size = window_size
        self.samples = []

        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])

        # Weather encoding (validated)
        weather_mapping = {
            "Cloudy": 0,
            "Rainy": 1,
            "Storm": 2,
            "Sunny": 3,
        }
        df["weather_enc"] = df["weather"].map(weather_mapping)

        if df["weather_enc"].isnull().any():
            raise ValueError("Unknown weather category found!")

        grouped = df.groupby(["store_id", "product_id"])

        for (_, product_id), group in grouped:
            group = group.sort_values("date")

            demand = group["daily_demand"].values
            price = group["price"].values
            holiday = group["holiday_flag"].values
            weather = group["weather_enc"].values

            for i in range(len(demand) - window_size):
                x = [
                    [
                        demand[j],
                        price[j],
                        holiday[j],
                        weather[j],
                    ]
                    for j in range(i, i + window_size)
                ]
                y = demand[i + window_size]

                # Store product_id per sample (CRITICAL for GNN)
                self.samples.append((x, y, product_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, product_id = self.samples[idx]

        x = torch.tensor(x, dtype=torch.float32)  # (W, 4)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y, product_id
