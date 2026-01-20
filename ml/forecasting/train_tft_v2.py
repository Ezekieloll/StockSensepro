import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from forecasting.dataset2 import DemandDatasetV2
from forecasting.tft_model import TemporalFusionTransformer


# ---------- Metrics ---------- #

def mae(y, yhat): return np.mean(np.abs(np.array(y) - np.array(yhat)))
def rmse(y, yhat): return np.sqrt(np.mean((np.array(y) - np.array(yhat))**2))
def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    mask = y != 0
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100
def wape(y, yhat):
    y = np.array(y)
    yhat = np.array(yhat)
    denom = y.sum()
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y - yhat)) / denom * 100



# ---------- Training ---------- #

def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = nn.MSELoss()(preds, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)


def eval_epoch(model, loader, device):
    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).cpu().numpy()
            y_all.extend(y.numpy())
            p_all.extend(preds)

    return mae(y_all, p_all), rmse(y_all, p_all), mape(y_all, p_all), wape(y_all, p_all)


# ---------- Main ---------- #

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DemandDatasetV2("data/processed2/train.csv", 17)
    val_ds = DemandDatasetV2("data/processed2/val.csv", 17)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=True, pin_memory=True)

    model = TemporalFusionTransformer(
        num_features=9,
        hidden_size=128
    ).to(device)

    print("Model device:", next(model.parameters()).device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 21):
        loss = train_epoch(model, train_loader, optimizer, device)
        mae_v, rmse_v, mape_v, wape_v = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"Train MSE: {loss:.4f} | "
            f"MAE: {mae_v:.4f} | "
            f"RMSE: {rmse_v:.4f} | "
            f"MAPE: {mape_v:.2f}% | "
            f"WAPE: {wape_v:.2f}%"
        )


if __name__ == "__main__":
    main()
