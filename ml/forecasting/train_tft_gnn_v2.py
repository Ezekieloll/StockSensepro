"""
TFT + GNN Training Script V2

This script properly integrates the original TFT model with GNN.
Based on claude_code.py with proper GNN integration.

Key improvements:
1. Uses the original tested TFT model
2. Wraps it with GNN enrichment
3. Compares TFT vs TFT+GNN fairly
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import json

from forecasting.dataset_v3 import DemandDatasetV3, load_graph_data, collate_with_sku
from forecasting.tft_model import TemporalFusionTransformer
from forecasting.tft_gnn_wrapper import TFTWithGNNWrapper, TFTEnsemble


# ==========================================
# METRICS (from claude_code.py)
# ==========================================

def mae(y, yhat):
    return np.mean(np.abs(np.array(y) - np.array(yhat)))


def rmse(y, yhat):
    return np.sqrt(np.mean((np.array(y) - np.array(yhat))**2))


def mape(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
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


def bias(y, yhat):
    """Forecast bias - detect over/under forecasting"""
    return np.mean(np.array(yhat) - np.array(y))


def forecast_accuracy(y, yhat):
    """100 - MAPE for easier interpretation"""
    return 100 - mape(y, yhat)


# ==========================================
# LOSS FUNCTIONS (from claude_code.py)
# ==========================================

class HuberLoss(nn.Module):
    """Huber loss - more robust to outliers than MSE"""
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
    """
    Quantile loss - penalizes under-forecasting more
    Better for inventory (stockouts worse than overstock)
    """
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
# EARLY STOPPING (from claude_code.py)
# ==========================================

class EarlyStopping:
    """Stop training when validation loss doesn't improve"""
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

def train_epoch(model, loader, optimizer, criterion, device, adj=None, clip_grad=1.0, use_gnn=True):
    """Train one epoch."""
    model.train()
    losses = []

    for batch in loader:
        x, y, sku_indices = batch
        x = x.to(device)
        y = y.to(device)
        sku_indices = sku_indices.to(device)

        optimizer.zero_grad()
        
        # Forward pass
        if use_gnn and adj is not None:
            preds = model(x, sku_indices=sku_indices, adj=adj)
        else:
            # Use original TFT forward (without GNN)
            if hasattr(model, 'tft'):
                preds = model.tft(x)
            else:
                preds = model(x)
        
        loss = criterion(preds, y)
        loss.backward()
        
        # Gradient clipping (from claude_code.py)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def eval_epoch(model, loader, device, adj=None, use_gnn=True):
    """Evaluate with all metrics."""
    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for batch in loader:
            x, y, sku_indices = batch
            x = x.to(device)
            sku_indices = sku_indices.to(device)
            
            # Forward pass
            if use_gnn and adj is not None:
                preds = model(x, sku_indices=sku_indices, adj=adj)
            else:
                if hasattr(model, 'tft'):
                    preds = model.tft(x)
                else:
                    preds = model(x)
            
            preds = preds.cpu().numpy()
            y_all.extend(y.numpy())
            p_all.extend(preds)

    metrics = {
        'mae': mae(y_all, p_all),
        'rmse': rmse(y_all, p_all),
        'mape': mape(y_all, p_all),
        'wape': wape(y_all, p_all),
        'bias': bias(y_all, p_all),
        'forecast_accuracy': forecast_accuracy(y_all, p_all)
    }
    
    return metrics


# ==========================================
# MAIN TRAINING LOOP
# ==========================================

def main():
    # Set random seeds (from claude_code.py)
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # ==========================================
    # HYPERPARAMETERS (from claude_code.py)
    # ==========================================
    WINDOW_SIZE = 30          # Same as claude_code.py
    NUM_FEATURES = 9          # Same features
    HIDDEN_SIZE = 128         # Same as claude_code.py
    LSTM_LAYERS = 2           # Same
    ATTENTION_HEADS = 4       # Same
    NUM_PRODUCTS = 240
    
    BATCH_SIZE = 256          # Same
    LEARNING_RATE = 0.0003    # Same as claude_code.py
    WEIGHT_DECAY = 1e-4       # Same
    EPOCHS = 50               # Same
    CLIP_GRAD = 1.0           # Same
    PATIENCE = 15             # Same
    
    USE_GNN = True            # Toggle GNN
    USE_QUANTILE_LOSS = True  # Penalize under-forecasting (fix negative bias)
    MODEL_TYPE = "wrapper"    # "wrapper" or "ensemble"
    
    # ==========================================
    # LOAD GRAPH DATA
    # ==========================================
    print("\n🕸️  Loading graph data...")
    try:
        adj, sku_to_idx, idx_to_sku = load_graph_data("models/gnn")
        adj = adj.to(device)
        print(f"✅ Graph loaded: {adj.shape[0]} products, {int((adj > 0).sum())} edges")
    except Exception as e:
        print(f"⚠️  Could not load graph: {e}")
        print("   Running without GNN...")
        adj = None
        sku_to_idx = {}
        USE_GNN = False
    
    # ==========================================
    # DATA LOADING
    # ==========================================
    print("\n📂 Loading datasets...")
    train_ds = DemandDatasetV3(
        "data/processed2/train.csv", 
        WINDOW_SIZE,
        sku_to_idx=sku_to_idx
    )
    val_ds = DemandDatasetV3(
        "data/processed2/val.csv", 
        WINDOW_SIZE,
        sku_to_idx=sku_to_idx
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_with_sku,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_with_sku,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"✅ Train samples: {len(train_ds):,}")
    print(f"✅ Val samples: {len(val_ds):,}")
    
    # ==========================================
    # MODEL
    # ==========================================
    print(f"\n🤖 Building model...")
    
    if MODEL_TYPE == "wrapper":
        model = TFTWithGNNWrapper(
            num_features=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            attention_heads=ATTENTION_HEADS,
            num_products=NUM_PRODUCTS,
            use_gnn=USE_GNN
        ).to(device)
    elif MODEL_TYPE == "ensemble":
        model = TFTEnsemble(
            num_features=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            attention_heads=ATTENTION_HEADS,
            num_products=NUM_PRODUCTS
        ).to(device)
    else:
        # Fallback to original TFT
        model = TemporalFusionTransformer(
            num_features=NUM_FEATURES,
            hidden_size=HIDDEN_SIZE,
            lstm_layers=LSTM_LAYERS,
            attention_heads=ATTENTION_HEADS
        ).to(device)
        USE_GNN = False
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model: {model.__class__.__name__}")
    print(f"✅ Parameters: {total_params:,}")
    print(f"✅ GNN enabled: {USE_GNN}")
    
    # ==========================================
    # OPTIMIZER & SCHEDULER (from claude_code.py)
    # ==========================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )
    
    # Loss function
    if USE_QUANTILE_LOSS:
        criterion = QuantileLoss(quantile=0.6)
        print("✅ Using QuantileLoss (bias toward higher predictions)")
    else:
        criterion = HuberLoss(delta=1.0)
        print("✅ Using HuberLoss")
    
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0.001)
    
    # ==========================================
    # TRACKING
    # ==========================================
    best_wape = float('inf')
    best_epoch = 0
    history = {
        'train_loss': [],
        'val_mape': [],
        'val_wape': [],
        'val_bias': []
    }
    
    Path("models").mkdir(exist_ok=True)
    
    # ==========================================
    # TRAINING LOOP
    # ==========================================
    print("\n" + "="*80)
    print(f"🚀 STARTING TRAINING (Model={MODEL_TYPE}, GNN={'ON' if USE_GNN else 'OFF'})")
    print("="*80)
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, 
            device, adj, CLIP_GRAD, USE_GNN
        )
        
        # Validate
        val_metrics = eval_epoch(model, val_loader, device, adj, USE_GNN)
        
        # Update scheduler
        scheduler.step(val_metrics['wape'])
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['val_mape'].append(val_metrics['mape'])
        history['val_wape'].append(val_metrics['wape'])
        history['val_bias'].append(val_metrics['bias'])
        
        # Print metrics
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
        
        # Save best model
        if val_metrics['wape'] < best_wape:
            best_wape = val_metrics['wape']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'history': history,
                'config': {
                    'hidden_size': HIDDEN_SIZE,
                    'lstm_layers': LSTM_LAYERS,
                    'attention_heads': ATTENTION_HEADS,
                    'use_gnn': USE_GNN,
                    'model_type': MODEL_TYPE
                }
            }, 'models/best_tft_gnn_v2.pt')
            print(f"   ✅ New best model! WAPE: {best_wape:.2f}%")
        
        # Early stopping check
        if early_stopping(val_metrics['wape'], epoch):
            break
    
    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    print(f"Best WAPE: {best_wape:.2f}% at epoch {best_epoch}")
    
    # Load best model
    checkpoint = torch.load('models/best_tft_gnn_v2.pt', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation WITH GNN
    final_metrics = eval_epoch(model, val_loader, device, adj, use_gnn=True)
    
    print(f"\n📊 FINAL VALIDATION METRICS ({MODEL_TYPE.upper()}):")
    print(f"   MAE:               {final_metrics['mae']:.2f} units")
    print(f"   RMSE:              {final_metrics['rmse']:.2f} units")
    print(f"   MAPE:              {final_metrics['mape']:.2f}%")
    print(f"   WAPE:              {final_metrics['wape']:.2f}%")
    print(f"   Forecast Accuracy: {final_metrics['forecast_accuracy']:.2f}%")
    print(f"   Bias:              {final_metrics['bias']:.2f} units")
    
    # Compare GNN vs No-GNN
    if USE_GNN and adj is not None:
        print("\n🔬 Comparing WITH GNN vs WITHOUT GNN...")
        no_gnn_metrics = eval_epoch(model, val_loader, device, adj, use_gnn=False)
        
        print(f"\n   {'Metric':<20} {'With GNN':<15} {'Without GNN':<15} {'Δ':>10}")
        print("   " + "-"*60)
        for metric in ['mae', 'mape', 'wape']:
            with_gnn = final_metrics[metric]
            without_gnn = no_gnn_metrics[metric]
            diff = without_gnn - with_gnn
            symbol = "✅" if diff > 0 else "❌"
            print(f"   {metric.upper():<20} {with_gnn:<15.2f} {without_gnn:<15.2f} {diff:+10.2f} {symbol}")
    
    # Interpretation
    print("\n💡 INTERPRETATION:")
    if final_metrics['wape'] < 20:
        print("   ✅ Excellent forecasting accuracy!")
    elif final_metrics['wape'] < 30:
        print("   ✓ Good forecasting accuracy")
    elif final_metrics['wape'] < 40:
        print("   ⚠️  Acceptable but needs improvement")
    else:
        print("   ❌ Poor accuracy - check data and features")
    
    if abs(final_metrics['bias']) < 0.5:
        print("   ✅ Low bias - balanced forecasting")
    elif final_metrics['bias'] > 0.5:
        print("   ⚠️  Over-forecasting bias (excess inventory risk)")
    else:
        print("   ⚠️  Under-forecasting bias (stockout risk)")
    
    # Save results
    with open('models/tft_gnn_v2_results.json', 'w') as f:
        json.dump({
            'final_metrics': {k: float(v) if not np.isnan(v) else None 
                            for k, v in final_metrics.items()},
            'best_epoch': best_epoch,
            'best_wape': float(best_wape),
            'use_gnn': USE_GNN,
            'model_type': MODEL_TYPE,
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ Results saved to models/tft_gnn_v2_results.json")
    
    return model, final_metrics


if __name__ == "__main__":
    model, metrics = main()
