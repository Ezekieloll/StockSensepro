import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

from forecasting.dataset import DemandDataset
from forecasting.transformer_model import TFTStyleTransformer


# --------------------
# Metrics
# --------------------
def rmse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# --------------------
# Training
# --------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, device):
    model.eval()
    preds_all = []
    y_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x)

            preds_all.extend(preds.cpu().numpy())
            y_all.extend(y.cpu().numpy())

    return rmse(y_all, preds_all), mape(y_all, preds_all)


# --------------------
# Main
# --------------------
def main():
    device = torch.device("cpu")

    train_ds = DemandDataset("data/processed/train.csv", window_size=14)
    val_ds = DemandDataset("data/processed/val.csv", window_size=14)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = TFTStyleTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = 5

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_rmse, val_mape = eval_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val RMSE: {val_rmse:.4f} | "
            f"Val MAPE: {val_mape:.2f}%"
        )


if __name__ == "__main__":
    main()
