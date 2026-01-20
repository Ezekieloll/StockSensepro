import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from forecasting.dataset import DemandDataset
from forecasting.lstm_gnn_model import LSTMGNNModel
from gnn.graph_utils import build_graph_tensors


def compute_wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom * 100


def main():
    device = torch.device("cpu")

    # Load graph
    nodes, node_to_idx, adj = build_graph_tensors(
        "data/raw/categories_products.csv"
    )
    adj = adj.to(device)

    # Model (structure only; consistent with other analyses)
    model = LSTMGNNModel(input_size=4, hidden_size=32).to(device)
    model.eval()

    # Validation dataset
    val_ds = DemandDataset("data/processed/val.csv", window_size=17)

    per_product_true = defaultdict(list)
    per_product_pred = defaultdict(list)

    with torch.no_grad():
        for x, y, product_id in val_ds:
            x = x.unsqueeze(0).to(device)

            sku_idx = torch.tensor(
                [node_to_idx[product_id]],
                device=device,
                dtype=torch.long,
            )

            pred = model(x, sku_idx, adj).item()

            per_product_true[product_id].append(y.item())
            per_product_pred[product_id].append(pred)

    records = []
    for pid in per_product_true:
        records.append(
            {
                "product_id": pid,
                "avg_demand": np.mean(per_product_true[pid]),
                "wape": compute_wape(
                    per_product_true[pid],
                    per_product_pred[pid],
                ),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv("analysis/wape_per_product.csv", index=False)

    print("✅ Saved: analysis/wape_per_product.csv")
    print(df.describe())


if __name__ == "__main__":
    main()
