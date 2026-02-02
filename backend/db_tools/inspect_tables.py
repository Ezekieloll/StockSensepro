from app.database import SessionLocal
from app.models.transaction import Transaction
from app.models.inventory import Inventory
from app.models.rebalancing_plan import RebalancingPlan
from app.models.user import User
from sqlalchemy import func, distinct
import pandas as pd

db = SessionLocal()

print("=" * 80)
print("TRANSACTIONS TABLE - First 100 Rows")
print("=" * 80)
transactions = db.query(Transaction).limit(100).all()
if transactions:
    tx_data = []
    for tx in transactions[:20]:  # Show first 20
        tx_data.append({
            'id': tx.id,
            'date': tx.date,
            'store_id': tx.store_id,
            'product_id': tx.product_id,
            'event_type': tx.event_type,
            'quantity': tx.quantity,
            'source': tx.source,
            'destination': tx.destination,
            'price': tx.price,
        })
    df = pd.DataFrame(tx_data)
    print(df.to_string())
    print(f"\nTotal transactions in DB: {db.query(Transaction).count()}")
else:
    print("No transactions found")

print("\n" + "=" * 80)
print("INVENTORY TABLE - All Rows")
print("=" * 80)
inventory = db.query(Inventory).limit(100).all()
if inventory:
    inv_data = []
    for inv in inventory[:30]:  # Show first 30
        inv_data.append({
            'id': inv.id,
            'sku': inv.sku,
            'store_id': inv.store_id,
            'quantity': inv.quantity,
            'last_updated': inv.last_updated,
        })
    df = pd.DataFrame(inv_data)
    print(df.to_string())
    print(f"\nTotal inventory records: {db.query(Inventory).count()}")
    
    # Show inventory summary by store
    print("\n--- Inventory Summary by Store ---")
    for store in ['Store_1', 'Store_2', 'Store_3']:
        count = db.query(Inventory).filter(Inventory.store_id == store).count()
        total_qty = db.query(func.sum(Inventory.quantity)).filter(Inventory.store_id == store).scalar() or 0
        print(f"{store}: {count} SKUs, Total Qty: {total_qty:.0f}")
else:
    print("No inventory found")

print("\n" + "=" * 80)
print("REBALANCING_PLAN TABLE - All Rows")
print("=" * 80)
rebalancing = db.query(RebalancingPlan).limit(100).all()
if rebalancing:
    rb_data = []
    for rb in rebalancing[:20]:
        rb_data.append({
            'id': rb.id,
            'sku': rb.sku,
            'from_store': rb.from_store,
            'to_store': rb.to_store,
            'quantity': rb.quantity,
            'status': rb.status,
            'created_at': rb.created_at,
        })
    df = pd.DataFrame(rb_data)
    print(df.to_string())
    print(f"\nTotal rebalancing plans: {db.query(RebalancingPlan).count()}")
else:
    print("No rebalancing plans found")

print("\n" + "=" * 80)
print("USERS TABLE - All Rows")
print("=" * 80)
users = db.query(User).all()
if users:
    user_data = []
    for u in users:
        user_data.append({
            'id': u.id,
            'name': u.name,
            'email': u.email,
            'role': u.role,
            'created_at': u.created_at,
        })
    df = pd.DataFrame(user_data)
    print(df.to_string())
else:
    print("No users found")

print("\n" + "=" * 80)
print("TRANSACTION ANALYSIS - Restock Events")
print("=" * 80)
restocks = db.query(Transaction).filter(Transaction.event_type == 'restock').limit(50).all()
if restocks:
    print(f"\nTotal restock events: {db.query(Transaction).filter(Transaction.event_type == 'restock').count()}")
    print("\nSample restock events:")
    rs_data = []
    for rs in restocks[:15]:
        rs_data.append({
            'date': rs.date,
            'store_id': rs.store_id,
            'product_id': rs.product_id,
            'quantity': rs.quantity,
            'source': rs.source,
            'destination': rs.destination,
            'on_hand_before': rs.on_hand_before,
            'on_hand_after': rs.on_hand_after,
        })
    df = pd.DataFrame(rs_data)
    print(df.to_string())
    
    # Analyze restock sources
    print("\n--- Restock Sources Distribution ---")
    sources = db.query(Transaction.source, func.count(Transaction.id)).filter(
        Transaction.event_type == 'restock'
    ).group_by(Transaction.source).all()
    for source, count in sources:
        print(f"{source}: {count} restock events")
else:
    print("No restock events found")

print("\n" + "=" * 80)
print("PRICE ANALYSIS - Average Prices per SKU")
print("=" * 80)
# Get average prices for top 20 SKUs
price_analysis = db.query(
    Transaction.product_id,
    func.avg(Transaction.price).label('avg_price'),
    func.min(Transaction.price).label('min_price'),
    func.max(Transaction.price).label('max_price'),
    func.count(Transaction.id).label('transactions')
).filter(
    Transaction.price.isnot(None),
    Transaction.event_type == 'sale'
).group_by(Transaction.product_id).limit(20).all()

if price_analysis:
    price_data = []
    for pa in price_analysis:
        price_data.append({
            'product_id': pa.product_id,
            'avg_price': f"${pa.avg_price:.2f}",
            'min_price': f"${pa.min_price:.2f}",
            'max_price': f"${pa.max_price:.2f}",
            'transactions': pa.transactions,
        })
    df = pd.DataFrame(price_data)
    print(df.to_string())
else:
    print("No price data found")

db.close()
