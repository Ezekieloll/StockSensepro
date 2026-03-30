"""
LSTM+GNN Training Script V2 (Upgraded)

Matches the TFT+GNN v2 pipeline for fair metric comparison:
- QuantileLoss (bias toward not under-forecasting)
- AdamW + ReduceLROnPlateau
- Early stopping + gradient clipping
- DemandDatasetV3 (window=30, O(1) indexing)
- Saves checkpoint + results JSON
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from forecasting.dataset_v3 import DemandDatasetV3, load_graph_data, collate_with_sku
from forecasting.lstm_gnn_model import LSTMGNNModel


# ==========================================
# METRICS
# ==========================================

def mae(y, yhat):
    return np.mean(np.abs(np.array(y) - np.array(yhat)))


def rmse(y, yhat):
    return np.sqrt(np.mean((np.array(y) - np.array(yhat)) ** 2))


def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    mask = y != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100


def wape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    denom = y.sum()
    if denom == 0:
        return np.nan
    return np.sum(np.abs(y - yhat)) / denom * 100


def bias(y, yhat):
    return np.mean(np.array(yhat) - np.array(y))


def forecast_accuracy(y, yhat):
    return 100 - wape(y, yhat)


# ==========================================
# LOSS FUNCTIONS
# ==========================================

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred, target):
        error = torch.abs(pred - target)
        is_small = error < self.delta
        squared_loss = 0.5 * error ** 2
        linear_loss = self.delta * (error - 0.5 * self.delta)
        return torch.mean(torch.where(is_small, squared_loss, linear_loss))


class QuantileLoss(nn.Module):
    """Penalizes under-forecasting more — better for inventory."""
    def __init__(self, quantile=0.6):
        super().__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        loss = torch.max(
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        return torch.mean(loss)


# ==========================================
# EARLY STOPPING
# ==========================================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠️  Early stopping triggered!")
                print(f"   Best epoch: {self.best_epoch} with WAPE: {self.best_loss:.2f}%")
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop


# ==========================================
# TRAINING FUNCTIONS
# ==========================================

def train_epoch(model, loader, optimizer, criterion, device, adj, clip_grad=1.0):
    model.train()
    losses = []

    for x, y, sku_idx in loader:
        x = x.to(device)
        y = y.to(device)
        sku_idx = sku_idx.to(device)

        optimizer.zero_grad()
        preds = model(x, sku_idx, adj)
        loss = criterion(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
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

    return {
        'mae': mae(y_all, p_all),
        'rmse': rmse(y_all, p_all),
        'mape': mape(y_all, p_all),
        'wape': wape(y_all, p_all),
        'bias': bias(y_all, p_all),
        'forecast_accuracy': forecast_accuracy(y_all, p_all),
    }


# ==========================================
# MAIN
# ==========================================

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Hyperparameters
    WINDOW_SIZE = 30
    BATCH_SIZE = 256
    LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    CLIP_GRAD = 1.0
    PATIENCE = 15
    USE_QUANTILE_LOSS = True

    # Graph
    print("\n🕸️  Loading graph data...")
    try:
        adj, sku_to_idx, idx_to_sku = load_graph_data("models/gnn")
        adj = adj.to(device)
        print(f"✅ Graph loaded: {adj.shape[0]} products, {int((adj > 0).sum())} edges")
    except Exception as e:
        print(f"❌ Could not load graph: {e}")
        raise

    # Data
    print("\n📂 Loading datasets...")
    train_ds = DemandDatasetV3("data/processed2/train.csv", window_size=WINDOW_SIZE, sku_to_idx=sku_to_idx)
    val_ds   = DemandDatasetV3("data/processed2/val.csv",   window_size=WINDOW_SIZE, sku_to_idx=sku_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_with_sku, pin_memory=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_with_sku, pin_memory=True, num_workers=0)

    print(f"✅ Train samples: {len(train_ds):,}")
    print(f"✅ Val samples:   {len(val_ds):,}")

    # Model
    model = LSTMGNNModel(input_size=9, hidden_size=128).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🤖 Model: {model.__class__.__name__} | Parameters: {total_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Loss
    if USE_QUANTILE_LOSS:
        criterion = QuantileLoss(quantile=0.6)
        print("✅ Using QuantileLoss(0.6)")
    else:
        criterion = HuberLoss(delta=1.0)
        print("✅ Using HuberLoss")

    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)

    best_wape = float('inf')
    best_epoch = 0
    history = {'train_loss': [], 'val_mape': [], 'val_wape': [], 'val_bias': []}

    Path("models").mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("🚀 STARTING LSTM+GNN V2 TRAINING")
    print("="*80)

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, adj, CLIP_GRAD)
        val_metrics = eval_epoch(model, val_loader, device, adj)

        scheduler.step(val_metrics['wape'])
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_mape'].append(val_metrics['mape'])
        history['val_wape'].append(val_metrics['wape'])
        history['val_bias'].append(val_metrics['bias'])

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Loss: {train_loss:6.4f} | "
            f"MAE: {val_metrics['mae']:5.2f} | "
            f"RMSE: {val_metrics['rmse']:5.2f} | "
            f"MAPE: {val_metrics['mape']:5.1f}% | "
            f"WAPE: {val_metrics['wape']:5.1f}% | "
            f"Bias: {val_metrics['bias']:6.2f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_metrics['wape'] < best_wape:
            best_wape = val_metrics['wape']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history,
                'config': {'hidden_size': 128, 'window_size': WINDOW_SIZE}
            }, 'models/best_lstm_gnn_v2.pt')
            print(f"   ✅ New best model! WAPE: {best_wape:.2f}%")

        if early_stopping(val_metrics['wape'], epoch):
            break

    # Final evaluation
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)

    checkpoint = torch.load('models/best_lstm_gnn_v2.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    final_metrics = eval_epoch(model, val_loader, device, adj)

    print(f"\n📊 FINAL METRICS:")
    print(f"   MAE:               {final_metrics['mae']:.2f}")
    print(f"   RMSE:              {final_metrics['rmse']:.2f}")
    print(f"   MAPE:              {final_metrics['mape']:.2f}%")
    print(f"   WAPE:              {final_metrics['wape']:.2f}%")
    print(f"   Forecast Accuracy: {final_metrics['forecast_accuracy']:.2f}%")
    print(f"   Bias:              {final_metrics['bias']:.2f}")

    with open('models/lstm_gnn_v2_final_results.json', 'w') as f:
        json.dump({
            'final_metrics': {k: float(v) if not np.isnan(v) else None
                              for k, v in final_metrics.items()},
            'best_epoch': best_epoch,
            'best_wape': float(best_wape),
            'model_type': 'LSTMGNNModel',
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }, f, indent=2)

    print("\n✅ Results saved to models/lstm_gnn_v2_final_results.json")
    return model, final_metrics


if __name__ == "__main__":
    main()
