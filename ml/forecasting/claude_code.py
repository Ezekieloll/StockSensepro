"""
Fixed TFT Training Script
Changes to reduce MAPE from 52% to <25%:
1. Better loss function (Huber instead of MSE)
2. Lower learning rate with scheduler
3. Gradient clipping
4. Early stopping
5. Better optimizer (AdamW)
6. Increased window size
7. Proper evaluation metrics
8. Model checkpointing
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from forecasting.dataset2 import DemandDatasetV2
from forecasting.tft_model import TemporalFusionTransformer


# ==========================================
# METRICS
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
# IMPROVED LOSS FUNCTIONS
# ==========================================

class HuberLoss(nn.Module):
    """
    Huber loss - more robust to outliers than MSE
    Good for demand forecasting with occasional spikes
    """
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
# EARLY STOPPING
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
# TRAINING WITH IMPROVEMENTS
# ==========================================

def train_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0):
    """
    Train one epoch with gradient clipping
    """
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        
        # CRITICAL: Gradient clipping prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        
        optimizer.step()
        losses.append(loss.item())

    return np.mean(losses)


def eval_epoch(model, loader, device):
    """
    Evaluate with all metrics
    """
    model.eval()
    y_all, p_all = [], []

    with torch.no_grad():
        for x, y in loader:
            preds = model(x.to(device)).cpu().numpy()
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
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # ==========================================
    # HYPERPARAMETERS (OPTIMIZED)
    # ==========================================
    WINDOW_SIZE = 30          # Increased from 17 to capture more patterns
    NUM_FEATURES = 9          # Your current features (add more if possible!)
    HIDDEN_SIZE = 128         # Good capacity
    LSTM_LAYERS = 2           # Was 1 in tft_model.py - update model!
    ATTENTION_HEADS = 4
    
    BATCH_SIZE = 256          # Your current (good)
    LEARNING_RATE = 0.0003    # REDUCED from 0.001 (critical!)
    WEIGHT_DECAY = 1e-4       # Regularization
    EPOCHS = 50               # Increased from 11
    CLIP_GRAD = 1.0
    PATIENCE = 15             # Early stopping patience
    
    # ==========================================
    # DATA LOADING
    # ==========================================
    print("\n📂 Loading datasets...")
    train_ds = DemandDatasetV2("data/processed2/train.csv", WINDOW_SIZE)
    val_ds = DemandDatasetV2("data/processed2/val.csv", WINDOW_SIZE)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        pin_memory=True,
        num_workers=0  # Set to 2-4 if using CPU
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True,
        num_workers=0
    )
    
    print(f"✅ Train samples: {len(train_ds):,}")
    print(f"✅ Val samples: {len(val_ds):,}")
    
    # ==========================================
    # MODEL
    # ==========================================
    print(f"\n🤖 Building model...")
    model = TemporalFusionTransformer(
        num_features=NUM_FEATURES,
        hidden_size=HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,        # Make sure your model supports this!
        attention_heads=ATTENTION_HEADS,                 # Make sure your model supports this!
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model parameters: {total_params:,}")
    print(f"✅ Model device: {next(model.parameters()).device}")
    
    # ==========================================
    # OPTIMIZER & SCHEDULER (IMPROVED)
    # ==========================================
    optimizer = torch.optim.AdamW(      # AdamW better than Adam
        model.parameters(),
        lr=LEARNING_RATE,               # Lower LR
        weight_decay=WEIGHT_DECAY       # L2 regularization
    )
    
    # Learning rate scheduler - reduces LR when stuck
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,         # Reduce LR by 50% when plateau
        patience=5,         # Wait 5 epochs before reducing
        verbose=True,
        min_lr=1e-6
    )
    
    # ==========================================
    # LOSS FUNCTION (IMPROVED)
    # ==========================================
    # Option 1: Huber Loss (recommended)
    criterion = HuberLoss(delta=1.0)
    
    # Option 2: Quantile Loss (if stockouts are critical)
    # criterion = QuantileLoss(quantile=0.6)
    
    # Option 3: Standard MSE (your current - not recommended)
    # criterion = nn.MSELoss()
    
    print(f"✅ Loss function: {criterion.__class__.__name__}")
    
    # ==========================================
    # EARLY STOPPING
    # ==========================================
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
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # ==========================================
    # TRAINING LOOP
    # ==========================================
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print("="*80)
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, CLIP_GRAD
        )
        
        # Validate
        val_metrics = eval_epoch(model, val_loader, device)
        
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
                'history': history
            }, 'models/best_tft_model.pt')
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
    checkpoint = torch.load('models/best_tft_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final validation
    final_metrics = eval_epoch(model, val_loader, device)
    
    print("\n📊 FINAL VALIDATION METRICS:")
    print(f"   MAE:               {final_metrics['mae']:.2f} units")
    print(f"   RMSE:              {final_metrics['rmse']:.2f} units")
    print(f"   MAPE:              {final_metrics['mape']:.2f}%")
    print(f"   WAPE:              {final_metrics['wape']:.2f}%")
    print(f"   Forecast Accuracy: {final_metrics['forecast_accuracy']:.2f}%")
    print(f"   Bias:              {final_metrics['bias']:.2f} units")
    
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
    
    print("\n" + "="*80)
    
    # Save final results
    import json
    with open('models/tft_final_results.json', 'w') as f:
        json.dump({
            'final_metrics': {k: float(v) if not np.isnan(v) else None 
                            for k, v in final_metrics.items()},
            'best_epoch': best_epoch,
            'best_wape': float(best_wape),
            'history': {k: [float(x) for x in v] for k, v in history.items()}
        }, f, indent=2)
    
    print("✅ Results saved to models/tft_final_results.json")
    
    return model, final_metrics


if __name__ == "__main__":
    model, metrics = main()