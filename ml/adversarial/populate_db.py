"""
Populate adversarial_risk database table with real risk calculations.
Runs adversarial scenarios and writes results to PostgreSQL.
"""
import sys
import os

# Add parent directory (ml/) to path for imports
ml_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(os.path.dirname(ml_dir), 'backend')
sys.path.insert(0, ml_dir)
sys.path.insert(0, backend_dir)

import torch
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from adversarial.scenario_simulator import ScenarioSimulator
from adversarial.inventory_risk import InventoryRiskEvaluator

# Backend imports
from app.models.adversarial_risk import AdversarialRisk
from app.models.inventory import Inventory
from app.config import DATABASE_URL

# -----------------------------
# CONFIG
# -----------------------------

# Paths relative to ml/ directory
ML_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ML_BASE, "models", "best_tft_gnn_v2.pt")

# -----------------------------
# LOAD INVENTORY FROM DB
# -----------------------------

def load_inventory_from_db(db):
    """Load real inventory from database"""
    inventory_records = db.query(Inventory).all()
    
    inventory = defaultdict(dict)
    for record in inventory_records:
        inventory[record.sku][record.store_id] = record.quantity
    
    return inventory

# -----------------------------
# MAIN PIPELINE
# -----------------------------

def main():
    print("\n🔥 Adversarial Risk Calculator - Database Population")
    print(f"📊 Database: {DATABASE_URL}\n")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}\n")

    # Database setup
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    # Clear existing data
    print("🗑️  Clearing old adversarial risk data...")
    db.query(AdversarialRisk).delete()
    db.commit()

    # Load REAL inventory from database
    print("📦 Loading inventory from database...")
    inventory = load_inventory_from_db(db)
    
    if not inventory:
        print("⚠️  WARNING: No inventory found in database!")
        print("   Using default inventory values as fallback...")
        inventory = defaultdict(lambda: {
            "Store_1": 50,
            "Store_2": 40,
            "Store_3": 30
        })
    else:
        print(f"✅ Loaded inventory for {len(inventory)} SKUs\n")

    # Get ALL SKUs directly from database (transactions table)
    from app.models.transaction import Transaction
    print("📂 Loading SKU list from database...")
    all_skus = db.query(Transaction.product_id).distinct().order_by(Transaction.product_id).all()
    all_skus = [sku[0] for sku in all_skus]
    print(f"✅ Found {len(all_skus)} SKUs in database\n")

    # Load model
    print(f"🤖 Loading model: {MODEL_PATH}")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint and 'config' in checkpoint:
            # Reconstruct model from config
            from forecasting.tft_gnn_wrapper import TFTWithGNNWrapper
            config = checkpoint['config']
            
            model = TFTWithGNNWrapper(
                num_features=9,  # From dataset
                hidden_size=config['hidden_size'],
                lstm_layers=config['lstm_layers'],
                attention_heads=config['attention_heads'],
                num_products=240,
                use_gnn=config.get('use_gnn', True)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            # Load GNN adjacency matrix (same as 3D visualizer)
            adjacency_path = os.path.join(ML_BASE, "models", "gnn", "adjacency.pt")
            sku_to_idx_path = os.path.join(ML_BASE, "models", "gnn", "sku_to_idx.pt")
            
            if os.path.exists(adjacency_path) and os.path.exists(sku_to_idx_path):
                adjacency = torch.load(adjacency_path, weights_only=False).to(device)
                sku_to_idx = torch.load(sku_to_idx_path, weights_only=False)
                model.adjacency = adjacency
                print("✅ Model loaded with GNN adjacency matrix\n")
                # Store for later use
                graph_data = {"adjacency": adjacency, "sku_to_idx": sku_to_idx}
            else:
                print("⚠️  Adjacency matrix not found, GNN disabled\n")
                graph_data = None
                
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            model = checkpoint["model"]
            model.to(device)
            model.eval()
            print("✅ Model loaded\n")
        else:
            print("⚠️  Unsupported checkpoint format, using real demand estimates\n")
            model = None
            
    except Exception as e:
        print(f"⚠️  Model load failed: {e}")
        print("   Using real demand from transaction history\n")
        model = None

    simulator = ScenarioSimulator(model, device) if model else None
    risk_eval = InventoryRiskEvaluator()

    # We don't need historical samples - using real demand from DailyDemand table
    results = []
    
    # Store ID mapping (inventory uses Store_1, demand data uses S1)
    STORE_MAP = {
        "Store_1": "S1",
        "Store_2": "S2", 
        "Store_3": "S3"
    }
    
    # Calculate demand from daily_demand table for realistic estimates
    from app.models.transaction import DailyDemand
    from sqlalchemy import func
    
    print("📊 Calculating average demand from transaction history...")
    demand_cache = {}
    for sku in all_skus:
        for store_demand_id in ["S1", "S2", "S3"]:
            avg_demand = db.query(func.avg(DailyDemand.total_quantity)).filter(
                DailyDemand.product_id == sku,
                DailyDemand.store_id == store_demand_id
            ).scalar()
            if avg_demand and avg_demand > 0:
                # Store with S1, S2, S3 format (matches inventory table)
                demand_cache[(sku, store_demand_id)] = float(avg_demand)
    
    print(f"✅ Found demand data for {len(demand_cache)} SKU-store combinations")
    print(f"📋 Sample cache keys: {list(demand_cache.keys())[:5]}\n")
    
    for sku in tqdm(all_skus, desc="Running adversarial tests"):
        # Get inventory for this SKU (or use defaults if not in DB)
        if sku in inventory:
            store_inventory = inventory[sku]
        else:
            # Default inventory if not in DB
            store_inventory = {
                "Store_1": 50,
                "Store_2": 40,
                "Store_3": 30
            }

        # Evaluate risk per store
        for store_id, inv_units in store_inventory.items():
            # Map Store_1 → S1 for demand cache lookup
            demand_store_id = STORE_MAP.get(store_id, store_id)
            # Use real demand average as baseline
            store_baseline = demand_cache.get((sku, demand_store_id), 5.0)
            # Worst-case: 10x spike for adversarial scenario (3x was too conservative given high inventory)
            store_worst = store_baseline * 10.0
            
            risk = risk_eval.evaluate(
                baseline_demand=store_baseline,
                worst_case_demand=store_worst,
                inventory_level=inv_units
            )

            # Create database record (store with S1/S2/S3 format)
            record = AdversarialRisk(
                sku=sku,
                store_id=demand_store_id,  # Use S1, S2, S3 format
                baseline_demand=store_baseline,
                worst_case_demand=store_worst,
                severity=risk["severity"],
                days_of_cover=risk["days_of_cover"],
                risk_score=risk["risk_score"],
                stockout=risk["stockout"]
            )
            db.add(record)

            results.append({
                "sku": sku,
                "store": demand_store_id,
                "baseline": store_baseline,
                "worst_case": store_worst,
                "stockout": risk["stockout"],
                "risk_score": risk["risk_score"]
            })

    # Commit to database
    print("\n💾 Writing to database...")
    db.commit()
    print(f"✅ Inserted {len(results)} adversarial risk records\n")

    # Summary stats
    df = pd.DataFrame(results)
    high_risk = df[df["stockout"] == True]
    
    print("📊 SUMMARY")
    print(f"   Total SKUs processed: {len(all_skus)}")
    print(f"   Total SKU-Store combinations: {len(df)}")
    print(f"   High-risk (stockout): {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")
    print(f"   Mean risk score: {df['risk_score'].mean():.3f}")
    print(f"   Max risk score: {df['risk_score'].max():.3f}")
    print(f"\n✅ Database population complete!\n")

    db.close()


if __name__ == "__main__":
    main()
