"""
Check transaction counts by date and their impact on inventory.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import func
from datetime import datetime, timedelta
from app.database import SessionLocal
from app.models.transaction import Transaction
from app.models.inventory import Inventory

def main():
    db = SessionLocal()
    
    print("\n📊 TRANSACTION ANALYSIS BY DATE\n")
    print("=" * 70)
    
    # Get last 7 days of transaction counts
    print("\n📅 TRANSACTIONS PER DAY (Last 7 days):")
    
    for i in range(7, 0, -1):
        target_date = datetime.now().date() - timedelta(days=i-1)
        
        total_txns = db.query(Transaction).filter(
            Transaction.date == target_date
        ).count()
        
        sales = db.query(Transaction).filter(
            Transaction.date == target_date,
            Transaction.event_type == 'sale'
        ).count()
        
        restocks = db.query(Transaction).filter(
            Transaction.date == target_date,
            Transaction.event_type.in_(['restock_request', 'restock_receipt'])
        ).count()
        
        if total_txns > 0:
            print(f"   {target_date}: {total_txns:,} total ({sales:,} sales, {restocks} restocks)")
    
    # Check specific Jan 30 data
    jan30 = datetime(2026, 1, 30).date()
    jan30_count = db.query(Transaction).filter(Transaction.date == jan30).count()
    
    print(f"\n📌 JANUARY 30, 2026 DETAILS:")
    print(f"   Total transactions: {jan30_count:,}")
    
    # Event type breakdown
    event_types = db.query(
        Transaction.event_type,
        func.count(Transaction.id).label('count')
    ).filter(
        Transaction.date == jan30
    ).group_by(Transaction.event_type).all()
    
    print(f"\n   Breakdown by type:")
    for event_type, count in event_types:
        print(f"      {event_type}: {count:,}")
    
    # Sample some Jan 30 transactions to see on_hand_after
    print(f"\n📋 SAMPLE JAN 30 TRANSACTIONS (showing on_hand_after):")
    samples = db.query(Transaction).filter(
        Transaction.date == jan30
    ).limit(10).all()
    
    for txn in samples:
        print(f"   {txn.product_id} @ {txn.store_id} | {txn.event_type} | Qty: {txn.quantity} | On-hand: {txn.on_hand_before} → {txn.on_hand_after}")
    
    print("\n" + "=" * 70 + "\n")
    
    db.close()

if __name__ == "__main__":
    main()
