"""
Quick check of inventory table contents.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func
from app.database import SessionLocal
from app.models.inventory import Inventory

def main():
    db = SessionLocal()
    
    print("\n📦 INVENTORY TABLE CHECK\n")
    print("=" * 60)
    
    # Count records
    total_records = db.query(Inventory).count()
    print(f"\n📊 Total inventory records: {total_records}")
    
    if total_records == 0:
        print("\n⚠️  INVENTORY TABLE IS EMPTY!")
        print("   Run: python scripts/populate_inventory.py\n")
        db.close()
        return
    
    # Summary stats
    total_units = db.query(func.sum(Inventory.quantity)).scalar() or 0
    avg_units = db.query(func.avg(Inventory.quantity)).scalar() or 0
    min_units = db.query(func.min(Inventory.quantity)).scalar() or 0
    max_units = db.query(func.max(Inventory.quantity)).scalar() or 0
    
    print(f"\n📈 STATISTICS:")
    print(f"   Total units: {total_units:,.0f}")
    print(f"   Average per SKU-Store: {avg_units:.1f}")
    print(f"   Min: {min_units:.0f}")
    print(f"   Max: {max_units:.0f}")
    
    # Store breakdown
    print(f"\n🏪 BY STORE:")
    stores = db.query(
        Inventory.store_id,
        func.count(Inventory.id).label('sku_count'),
        func.sum(Inventory.quantity).label('total_units')
    ).group_by(Inventory.store_id).all()
    
    for store_id, sku_count, total in stores:
        print(f"   {store_id}: {sku_count} SKUs, {total:,.0f} units")
    
    # Sample data
    print(f"\n📋 SAMPLE RECORDS (first 10):")
    print(f"   {'SKU':<20} {'Store':<8} {'Quantity':>10}")
    print(f"   {'-'*20} {'-'*8} {'-'*10}")
    
    samples = db.query(Inventory).limit(10).all()
    for inv in samples:
        print(f"   {inv.sku:<20} {inv.store_id:<8} {inv.quantity:>10.0f}")
    
    print("\n" + "=" * 60)
    print("✅ Inventory check complete!\n")
    
    db.close()

if __name__ == "__main__":
    main()
