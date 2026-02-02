"""
Check when inventory was last updated and if it's in sync with transactions.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func, desc
from datetime import datetime
from app.database import SessionLocal
from app.models.inventory import Inventory
from app.models.transaction import Transaction

def main():
    db = SessionLocal()
    
    print("\n📦 INVENTORY UPDATE STATUS\n")
    print("=" * 70)
    
    # Check most recent inventory update
    latest_inv = db.query(Inventory).order_by(desc(Inventory.last_updated)).first()
    if latest_inv:
        print(f"\n⏰ INVENTORY LAST UPDATED:")
        print(f"   {latest_inv.last_updated}")
        print(f"   SKU: {latest_inv.sku} at {latest_inv.store_id}")
        
        days_old = (datetime.now() - latest_inv.last_updated.replace(tzinfo=None)).days
        print(f"   📅 Age: {days_old} days old")
        
        if days_old > 1:
            print(f"   ⚠️  WARNING: Inventory data is stale!")
    
    # Check most recent transaction
    latest_txn = db.query(Transaction).order_by(desc(Transaction.date)).first()
    if latest_txn:
        print(f"\n🔄 LATEST TRANSACTION:")
        print(f"   Date: {latest_txn.date}")
        print(f"   Type: {latest_txn.event_type}")
        print(f"   SKU: {latest_txn.product_id} at {latest_txn.store_id}")
    
    # Check for recent restock_receipt transactions
    recent_restocks = db.query(Transaction).filter(
        Transaction.event_type == 'restock_receipt'
    ).order_by(desc(Transaction.date)).limit(5).all()
    
    if recent_restocks:
        print(f"\n📦 RECENT RESTOCK RECEIPTS (Last 5):")
        for txn in recent_restocks:
            print(f"   {txn.date} | {txn.product_id} @ {txn.store_id} | Qty: {txn.quantity} | On-hand after: {txn.on_hand_after}")
    
    # Check if any inventory records were updated after latest transaction
    if latest_txn and latest_inv:
        inv_date = latest_inv.last_updated.replace(tzinfo=None).date()
        txn_date = latest_txn.date
        inv_newer = inv_date >= txn_date
        print(f"\n🔍 SYNC STATUS:")
        print(f"   Inventory date: {inv_date}")
        print(f"   Transaction date: {txn_date}")
        print(f"   Inventory synced: {'✅ YES' if inv_newer else '❌ NO'}")
        
        if not inv_newer:
            print(f"\n   ⚠️  PROBLEM: Inventory is NOT being updated when transactions occur!")
            print(f"   Latest transaction: {txn_date}")
            print(f"   Latest inventory update: {inv_date}")
    
    print("\n" + "=" * 70 + "\n")
    
    db.close()

if __name__ == "__main__":
    main()
