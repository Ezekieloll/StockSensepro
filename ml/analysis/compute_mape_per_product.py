from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from forecasting.dataset_v3 import DemandDatasetV3, collate_with_sku, load_graph_data
from forecasting.lstm_gnn_model import LSTMGNNModel


def compute_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def main():
    base_dir = Path(__file__).resolve().parent.parent
    output_dir = base_dir / "analysis" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    val_csv = base_dir / "data" / "processed2" / "val.csv"
    model_path = base_dir / "models" / "best_lstm_gnn_v2.pt"
    graph_dir = base_dir / "models" / "gnn"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adj, sku_to_idx, idx_to_sku = load_graph_data(str(graph_dir))
    adj = adj.to(device)

    model = LSTMGNNModel(input_size=9, hidden_size=128).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_ds = DemandDatasetV3(str(val_csv), window_size=30, sku_to_idx=sku_to_idx)
    val_loader = DataLoader(
        val_ds,
        batch_size=512,
        shuffle=False,
        collate_fn=collate_with_sku,
        num_workers=0,
    )

    # Storage
    per_product_true = defaultdict(list)
    per_product_pred = defaultdict(list)

    with torch.no_grad():
        for x, y, sku_idx in val_loader:
            x = x.to(device)
            y = y.to(device)
            sku_idx = sku_idx.to(device)

            preds = model(x, sku_idx, adj)

            for i in range(x.size(0)):
                sku_index = int(sku_idx[i].item())
                if sku_index < 0:
                    continue
                product_id = idx_to_sku[sku_index]
                per_product_true[product_id].append(float(y[i].item()))
                per_product_pred[product_id].append(float(preds[i].item()))

    # Compute MAPE per product
    records = []
    for pid in per_product_true:
        mape = compute_mape(
            per_product_true[pid],
            per_product_pred[pid],
        )
        if np.isnan(mape):
            continue
        records.append(
            {
                "product_id": pid,
                "mape": mape,
                "avg_demand": np.mean(per_product_true[pid]),
            }
        )

    df = pd.DataFrame(records)
    df = df.sort_values("mape", ascending=True).reset_index(drop=True)
    output_file = output_dir / "mape_per_product.csv"
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(df.describe())


if __name__ == "__main__":
    main()
