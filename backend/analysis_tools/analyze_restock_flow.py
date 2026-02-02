from app.database import SessionLocal
from app.models.transaction import Transaction
from sqlalchemy import func
import pandas as pd

db = SessionLocal()

print("=" * 100)
print("RESTOCK FLOW ANALYSIS")
print("=" * 100)

# Get all unique event types
print("\n1. ALL EVENT TYPES IN DATABASE:")
event_types = db.query(
    Transaction.event_type,
    func.count(Transaction.id).label('count')
).group_by(Transaction.event_type).all()

for event_type, count in event_types:
    percentage = (count / 837939) * 100
    print(f"   {event_type}: {count:,} ({percentage:.2f}%)")

# Get all restock-related transactions
print("\n" + "=" * 100)
print("2. ALL RESTOCK-RELATED TRANSACTIONS:")
print("=" * 100)

restock_requests = db.query(Transaction).filter(
    Transaction.event_type == 'restock_request'
).limit(50).all()

print(f"\nTotal 'restock_request' transactions: {db.query(Transaction).filter(Transaction.event_type == 'restock_request').count()}")

if restock_requests:
    print("\nSample restock_request transactions (first 20):")
    req_data = []
    for req in restock_requests[:20]:
        req_data.append({
            'id': req.id,
            'date': str(req.date),
            'store': req.store_id,
            'product': req.product_id,
            'qty': req.quantity,
            'source': req.source,
            'destination': req.destination,
            'on_hand_before': req.on_hand_before,
            'on_hand_after': req.on_hand_after,
        })
    df = pd.DataFrame(req_data)
    print(df.to_string())

# Check for 'restock' event type (delivery)
print("\n" + "=" * 100)
print("3. SEARCHING FOR 'restock' (DELIVERY) TRANSACTIONS:")
print("=" * 100)

restock_deliveries = db.query(Transaction).filter(
    Transaction.event_type == 'restock'
).limit(50).all()

restock_count = db.query(Transaction).filter(Transaction.event_type == 'restock').count()
print(f"\nTotal 'restock' (delivery) transactions: {restock_count}")

if restock_deliveries:
    print("\nSample restock delivery transactions (first 20):")
    del_data = []
    for delivery in restock_deliveries[:20]:
        del_data.append({
            'id': delivery.id,
            'date': str(delivery.date),
            'store': delivery.store_id,
            'product': delivery.product_id,
            'qty': delivery.quantity,
            'source': delivery.source,
            'destination': delivery.destination,
            'price': f"${delivery.price:.2f}" if delivery.price else None,
            'on_hand_before': delivery.on_hand_before,
            'on_hand_after': delivery.on_hand_after,
        })
    df = pd.DataFrame(del_data)
    print(df.to_string())
else:
    print("❌ NO 'restock' delivery transactions found!")

# Check for warehouse → store flow
print("\n" + "=" * 100)
print("4. WAREHOUSE → STORE TRANSACTIONS:")
print("=" * 100)

warehouse_to_store = db.query(Transaction).filter(
    Transaction.source == 'warehouse'
).limit(50).all()

wh_count = db.query(Transaction).filter(Transaction.source == 'warehouse').count()
print(f"\nTotal transactions FROM warehouse: {wh_count}")

if warehouse_to_store:
    print("\nSample warehouse → store transactions (first 20):")
    wh_data = []
    for wh in warehouse_to_store[:20]:
        wh_data.append({
            'id': wh.id,
            'date': str(wh.date),
            'event': wh.event_type,
            'store': wh.store_id,
            'product': wh.product_id,
            'qty': wh.quantity,
            'source': wh.source,
            'destination': wh.destination,
            'on_hand_before': wh.on_hand_before,
            'on_hand_after': wh.on_hand_after,
        })
    df = pd.DataFrame(wh_data)
    print(df.to_string())
else:
    print("❌ NO transactions from warehouse found!")

# Analyze restock_request patterns
print("\n" + "=" * 100)
print("5. RESTOCK_REQUEST PATTERNS:")
print("=" * 100)

# Group by quantity
qty_distribution = db.query(
    Transaction.quantity,
    func.count(Transaction.id).label('count')
).filter(
    Transaction.event_type == 'restock_request'
).group_by(Transaction.quantity).all()

print("\nQuantity Distribution for restock_request:")
for qty, count in qty_distribution:
    print(f"   Quantity {qty}: {count:,} requests")

# Check if requests have matching deliveries
print("\n" + "=" * 100)
print("6. CHECKING REQUEST → DELIVERY MATCHING:")
print("=" * 100)

# Get a sample request
sample_request = db.query(Transaction).filter(
    Transaction.event_type == 'restock_request'
).first()

if sample_request:
    print(f"\nSample Request: ID={sample_request.id}, Date={sample_request.date}, "
          f"Store={sample_request.store_id}, Product={sample_request.product_id}, Qty={sample_request.quantity}")
    
    # Look for corresponding delivery (same product, store, date or later)
    possible_delivery = db.query(Transaction).filter(
        Transaction.product_id == sample_request.product_id,
        Transaction.store_id == sample_request.store_id,
        Transaction.date >= sample_request.date,
        Transaction.source == 'warehouse'
    ).first()
    
    if possible_delivery:
        print(f"\nPossible Matching Delivery: ID={possible_delivery.id}, Date={possible_delivery.date}, "
              f"Event={possible_delivery.event_type}, Qty={possible_delivery.quantity}")
    else:
        print("\n❌ NO matching delivery found for this request!")

db.close()
