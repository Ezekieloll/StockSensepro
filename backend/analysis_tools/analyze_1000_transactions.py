from app.database import SessionLocal
from app.models.transaction import Transaction
from sqlalchemy import func
import pandas as pd

db = SessionLocal()

print("=" * 100)
print("FIRST 1000 TRANSACTION RECORDS")
print("=" * 100)

# Get first 1000 transactions
transactions = db.query(Transaction).limit(1000).all()

print(f"\nTotal transactions fetched: {len(transactions)}")

# Convert to DataFrame for display
tx_data = []
for tx in transactions:
    tx_data.append({
        'id': tx.id,
        'date': str(tx.date),
        'store': tx.store_id,
        'product': tx.product_id,
        'category': tx.product_category,
        'event': tx.event_type,
        'qty': tx.quantity,
        'source': tx.source,
        'destination': tx.destination,
        'price': f"${tx.price:.2f}" if tx.price else None,
        'on_hand_before': tx.on_hand_before,
        'on_hand_after': tx.on_hand_after,
    })

df = pd.DataFrame(tx_data)

# Display all 1000 rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("\n" + "=" * 100)
print("ALL 1000 TRANSACTIONS:")
print("=" * 100)
print(df.to_string(index=True))

db.close()
