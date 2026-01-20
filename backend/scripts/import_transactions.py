"""
Import Transaction Data from CSV to PostgreSQL

This script:
1. Creates the transactions and daily_demand tables
2. Loads all 2023-2024 transaction data from CSV files
3. Pre-computes daily demand aggregates

Usage:
    cd c:\StockSense\backend
    python scripts/import_transactions.py
"""

import sys
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import DATABASE_URL
from app.models.transaction import Transaction, DailyDemand
from app.database import Base


def import_transactions():
    """Import transaction data from CSV files into PostgreSQL."""
    
    print("🔄 Connecting to database...")
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    
    # Create tables
    print("📊 Creating tables...")
    Base.metadata.create_all(engine)
    
    # Find CSV files - match actual filenames
    data_dir = Path(__file__).parent.parent.parent / "ml" / "data" / "raw"
    csv_files = list(data_dir.glob("transactions_3stores_*.csv"))
    
    if not csv_files:
        # Try alternate pattern
        csv_files = list(data_dir.glob("transactions_*.csv"))
    
    if not csv_files:
        print(f"❌ No transaction CSV files found in {data_dir}")
        return
    
    print(f"📂 Found {len(csv_files)} CSV files:")
    for f in csv_files:
        print(f"   - {f.name}")
    
    session = Session()
    
    # Check if data already exists
    existing_count = session.query(Transaction).count()
    if existing_count > 0:
        print(f"\n⚠️ Database already has {existing_count:,} transactions")
        response = input("   Clear existing data and reimport? (y/N): ")
        if response.lower() == 'y':
            print("   Clearing existing data...")
            session.query(DailyDemand).delete()
            session.query(Transaction).delete()
            session.commit()
        else:
            print("   Skipping import")
            session.close()
            return
    
    # Track daily demand for aggregation
    daily_demand = defaultdict(lambda: {"quantity": 0, "count": 0, "revenue": 0, "category": None, "prices": []})
    
    total_imported = 0
    batch_size = 10000
    batch = []
    
    for csv_file in sorted(csv_files):
        print(f"\n📥 Processing {csv_file.name}...")
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    # Parse date
                    date_str = row.get('date', '')
                    if ' ' in date_str:
                        date_str = date_str.split(' ')[0]
                    parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    
                    # Parse timestamp if available
                    timestamp = None
                    if row.get('timestamp'):
                        try:
                            timestamp = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    
                    # Create transaction with exact CSV column names
                    transaction = Transaction(
                        timestamp=timestamp,
                        date=parsed_date,
                        store_id=row.get('store_id', 'S1'),
                        product_id=row.get('product_id', ''),
                        product_category=row.get('product_category'),
                        event_type=row.get('event_type', 'sale'),
                        quantity=float(row.get('quantity', 1)),
                        on_hand_before=int(row['on_hand_before']) if row.get('on_hand_before') else None,
                        on_hand_after=int(row['on_hand_after']) if row.get('on_hand_after') else None,
                        source=row.get('source'),
                        destination=row.get('destination'),
                        price=float(row['price']) if row.get('price') else None,
                        holiday_flag=int(row['holiday_flag']) if row.get('holiday_flag') else None,
                        weather=row.get('weather'),
                        is_simulated=0
                    )
                    
                    batch.append(transaction)
                    
                    # Track daily demand (only for sales)
                    if row.get('event_type') == 'sale':
                        key = (parsed_date, row.get('store_id', 'S1'), row.get('product_id', ''))
                        daily_demand[key]["quantity"] += float(row.get('quantity', 1))
                        daily_demand[key]["count"] += 1
                        daily_demand[key]["category"] = row.get('product_category')
                        if row.get('price'):
                            price = float(row['price'])
                            daily_demand[key]["revenue"] += price * float(row.get('quantity', 1))
                            daily_demand[key]["prices"].append(price)
                    
                    # Batch insert
                    if len(batch) >= batch_size:
                        session.bulk_save_objects(batch)
                        session.commit()
                        total_imported += len(batch)
                        print(f"   Imported {total_imported:,} transactions...")
                        batch = []
                        
                except Exception as e:
                    print(f"   ⚠️ Error processing row: {e}")
                    continue
        
        # Final batch for this file
        if batch:
            session.bulk_save_objects(batch)
            session.commit()
            total_imported += len(batch)
            batch = []
    
    print(f"\n✅ Imported {total_imported:,} transactions")
    
    # Insert daily demand aggregates
    print("\n📊 Computing daily demand aggregates...")
    
    demand_records = []
    for (date, store_id, product_id), values in daily_demand.items():
        avg_price = sum(values["prices"]) / len(values["prices"]) if values["prices"] else None
        
        demand_records.append(DailyDemand(
            date=date,
            store_id=store_id,
            product_id=product_id,
            product_category=values["category"],
            total_quantity=values["quantity"],
            transaction_count=values["count"],
            total_revenue=values["revenue"] if values["revenue"] > 0 else None,
            avg_price=avg_price
        ))
    
    print(f"   Inserting {len(demand_records):,} daily demand records...")
    
    # Batch insert
    for i in range(0, len(demand_records), batch_size):
        session.bulk_save_objects(demand_records[i:i+batch_size])
        session.commit()
        if (i + batch_size) % 50000 == 0:
            print(f"   {i + batch_size:,} records...")
    
    print(f"✅ Created {len(demand_records):,} daily demand aggregates")
    
    # Get date range
    min_date = session.query(Transaction.date).order_by(Transaction.date.asc()).first()
    max_date = session.query(Transaction.date).order_by(Transaction.date.desc()).first()
    
    print(f"\n📅 Data range: {min_date[0]} to {max_date[0]}")
    
    # Get some stats
    total_stores = session.query(Transaction.store_id).distinct().count()
    total_products = session.query(Transaction.product_id).distinct().count()
    
    print(f"🏪 Stores: {total_stores}")
    print(f"📦 Products: {total_products}")
    
    session.close()
    print("\n🎉 Import complete!")


if __name__ == "__main__":
    import_transactions()
