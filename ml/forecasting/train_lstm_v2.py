import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from forecasting.dataset2 import DemandDatasetV2
from forecasting.lstm_model import LSTMBaseline


# ---------------- Metrics ---------------- #

def mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def wape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    denom = y_true.sum()
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denom * 100


# ---------------- Training ---------------- #

def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x).squeeze()
        loss = nn.MSELoss()(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def eval_epoch(model, loader, device):
    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            preds = model(x).squeeze().cpu().numpy()

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
   
    train_ds = DemandDatasetV2("data/processed2/train.csv", window_size=17)
    val_ds = DemandDatasetV2("data/processed2/val.csv", window_size=17)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)

    model = LSTMBaseline(
        input_size=9,
        hidden_size=128
    ).to(device)
    print("Model device:", next(model.parameters()).device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, 11):
        train_loss = train_epoch(
            model, train_loader, optimizer, device
        )

        val_mae, val_rmse, val_mape, val_wape = eval_epoch(
            model, val_loader, device
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
