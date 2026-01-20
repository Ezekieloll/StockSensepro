import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from forecasting.dataset2 import DemandDatasetV2
from forecasting.lstm_gnn_model import LSTMGNNModel


# ---------------- Metrics ---------------- #

def mae(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    return np.mean(np.abs(y - yhat))


def rmse(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    return np.sqrt(np.mean((y - yhat) ** 2))


def mape(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    mask = y != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100


def wape(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    denom = y.sum()
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y - yhat)) / denom * 100


# ---------------- Training ---------------- #

def train_epoch(model, loader, optimizer, device, adj):
    model.train()
    losses = []

    for x, y, sku_idx in loader:
        x = x.to(device)
        y = y.to(device)
        sku_idx = sku_idx.to(device)

        optimizer.zero_grad()
        preds = model(x, sku_idx, adj)
        loss = nn.MSELoss()(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def eval_epoch(model, loader, device, adj):
    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for x, y, sku_idx in loader:
            x = x.to(device)
            sku_idx = sku_idx.to(device)

            preds = model(x, sku_idx, adj).cpu().numpy()
            y_all.extend(y.numpy())
            p_all.extend(preds)

    return (
        mae(y_all, p_all),
        rmse(y_all, p_all),
        mape(y_all, p_all),
        wape(y_all, p_all),
    )


# ---------------- Main ---------------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load static graph
    adj = torch.load("data/processed2/adjacency.pt").to(device)
    sku_to_idx = torch.load("data/processed2/sku_to_idx.pt")

    # Dataset
    train_ds = DemandDatasetV2(
        "data/processed2/train.csv",
        window_size=17,
        sku_to_idx=sku_to_idx,
    )

    val_ds = DemandDatasetV2(
        "data/processed2/val.csv",
        window_size=17,
        sku_to_idx=sku_to_idx,
    )

    train_loader = DataLoader(
        train_ds, batch_size=128, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=128, shuffle=False, pin_memory=True
    )

    # Model
    model = LSTMGNNModel(
        input_size=9,
        hidden_size=128,
    ).to(device)

    print("Model device:", next(model.parameters()).device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(1, 21):
        train_loss = train_epoch(
            model, train_loader, optimizer, device, adj
        )

        val_mae, val_rmse, val_mape, val_wape = eval_epoch(
            model, val_loader, device, adj
        )

        print(
            f"Epoch {epoch} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MAE: {val_mae:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAPE: {val_mape:.2f}% | "
            f"Val WAPE: {val_wape:.2f}%"
        )


if __name__ == "__main__":
    main()
