import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from forecasting.dataset import DemandDataset
from forecasting.lstm_gnn_model import LSTMGNNModel
from gnn.graph_utils import build_graph_tensors


def compute_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    device = torch.device("cpu")

    # Load graph
    nodes, node_to_idx, adj = build_graph_tensors(
        "data/raw/categories_products.csv"
    )
    adj = adj.to(device)

    # Load trained model (retrain quickly or reuse weights if saved)
    model = LSTMGNNModel(input_size=4, hidden_size=32).to(device)
    model.eval()

    # Validation dataset
    val_ds = DemandDataset("data/processed/val.csv", window_size=17)

    # Storage
    per_product_true = defaultdict(list)
    per_product_pred = defaultdict(list)

    with torch.no_grad():
        for x, y, product_id in val_ds:
            x = x.unsqueeze(0).to(device)

            sku_idx = torch.tensor(
                [node_to_idx[product_id]], device=device
            )

            pred = model(x, sku_idx, adj).item()

            per_product_true[product_id].append(y.item())
            per_product_pred[product_id].append(pred)

    # Compute MAPE per product
    records = []
    for pid in per_product_true:
        mape = compute_mape(
            per_product_true[pid],
            per_product_pred[pid],
        )
        records.append(
            {
                "product_id": pid,
                "mape": mape,
                "avg_demand": np.mean(per_product_true[pid]),
            }
        )

    df = pd.DataFrame(records)
    df.to_csv("analysis/mape_per_product.csv", index=False)

    print("Saved per-product MAPE to analysis/mape_per_product.csv")
    print(df.describe())


if __name__ == "__main__":
    main()
