"""
Populate inventory table from transaction history.
Uses most recent on_hand_after values from transactions table.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import func
from app.database import SessionLocal
from app.models.transaction import Transaction
from app.models.inventory import Inventory

def main():
    db = SessionLocal()
    
    print("\n📦 Inventory Population from Transaction History\n")
    
    # Clear existing inventory
    print("🗑️  Clearing old inventory data...")
    db.query(Inventory).delete()
    db.commit()
    
    # Get unique product-store combinations
    print("🔍 Finding unique product-store combinations...")
    combos = db.query(
        Transaction.product_id,
        Transaction.store_id
    ).distinct().all()
    
    print(f"✅ Found {len(combos)} product-store combinations\n")
    
    created = 0
    no_data = 0
    
    print("📊 Calculating current inventory levels...\n")
    
    for product_id, store_id in combos:
        # Get most recent transaction for this product-store combo
        latest = db.query(Transaction).filter(
            Transaction.product_id == product_id,
            Transaction.store_id == store_id
        ).order_by(Transaction.date.desc()).first()
        
        if latest and latest.on_hand_after is not None:
            # Use actual on-hand quantity from latest transaction
            quantity = max(0, latest.on_hand_after)
            
            inventory = Inventory(
                sku=product_id,
                store_id=store_id,
                quantity=quantity
            )
            db.add(inventory)
            created += 1
        else:
            no_data += 1
    
    db.commit()
    
    print(f"✅ Created {created} inventory records")
    print(f"⚠️  Skipped {no_data} combinations (no on_hand data)")
    
    # Summary stats
    print("\n📊 INVENTORY SUMMARY")
    total_inventory = db.query(func.sum(Inventory.quantity)).scalar() or 0
    avg_inventory = db.query(func.avg(Inventory.quantity)).scalar() or 0
    min_inventory = db.query(func.min(Inventory.quantity)).scalar() or 0
    max_inventory = db.query(func.max(Inventory.quantity)).scalar() or 0
    
    print(f"   Total units across all stores: {total_inventory:,.0f}")
    print(f"   Average inventory per SKU-Store: {avg_inventory:.1f}")
    print(f"   Min inventory: {min_inventory:.0f}")
    print(f"   Max inventory: {max_inventory:.0f}")
    
    print("\n✅ Inventory population complete!\n")
    
    db.close()

if __name__ == "__main__":
    main()
