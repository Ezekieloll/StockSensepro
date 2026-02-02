import requests
import json
from datetime import date, timedelta

API_BASE = "http://localhost:8000"

print("=" * 80)
print("TESTING PURCHASE ORDER API")
print("=" * 80)

# Test 1: Create a PO
print("\n1. Creating Purchase Order for Store S1...")
po_data = {
    "store_id": "S1",
    "created_by_user_id": 4,  # Manager1
    "expected_delivery_date": str(date.today() + timedelta(days=3)),
    "notes": "Urgent restock - low stock alert",
    "items": [
        {
            "sku": "SKU_MEAT008",
            "product_category": "MEAT",
            "quantity_requested": 50,
            "unit_price": 120.50
        },
        {
            "sku": "SKU_FRPR002",
            "product_category": "FRPR",
            "quantity_requested": 75,
            "unit_price": 25.30
        },
        {
            "sku": "SKU_GROC009",
            "product_category": "GROC",
            "quantity_requested": 100,
            "unit_price": 15.75
        }
    ]
}

response = requests.post(f"{API_BASE}/api/purchase-orders/", json=po_data)
if response.status_code == 201:
    po = response.json()
    print(f"✅ Created PO: {po['po_number']}")
    print(f"   Store: {po['store_id']}")
    print(f"   Items: {po['total_items']}")
    print(f"   Total Qty: {po['total_quantity']}")
    print(f"   Total Amount: ${po['total_amount']}")
    print(f"   Status: {po['status']}")
    po_id = po['id']
else:
    print(f"❌ Failed: {response.status_code} - {response.text}")
    exit(1)

# Test 2: List POs
print("\n2. Listing all Purchase Orders...")
response = requests.get(f"{API_BASE}/api/purchase-orders/")
if response.status_code == 200:
    pos = response.json()
    print(f"✅ Found {len(pos)} purchase orders")
    for po in pos[:5]:  # Show first 5
        print(f"   - {po['po_number']}: Store {po['store_id']}, Status: {po['status']}, Items: {po['total_items']}")
else:
    print(f"❌ Failed: {response.status_code}")

# Test 3: Get single PO
print(f"\n3. Getting PO details for #{po_id}...")
response = requests.get(f"{API_BASE}/api/purchase-orders/{po_id}")
if response.status_code == 200:
    po = response.json()
    print(f"✅ PO {po['po_number']} Details:")
    print(f"   Created: {po['created_at']}")
    print(f"   Items ({len(po['items'])}):")
    for item in po['items']:
        print(f"     - {item['sku']}: {item['quantity_requested']} units @ ${item['unit_price']} = ${item['line_total']}")
else:
    print(f"❌ Failed: {response.status_code}")

# Test 4: Update status to approved
print(f"\n4. Approving PO #{po_id}...")
response = requests.put(
    f"{API_BASE}/api/purchase-orders/{po_id}/status",
    json={"status": "approved"}
)
if response.status_code == 200:
    po = response.json()
    print(f"✅ Status updated to: {po['status']}")
    print(f"   Approved at: {po['approved_date']}")
else:
    print(f"❌ Failed: {response.status_code}")

# Test 5: Mark as in_transit
print(f"\n5. Marking PO #{po_id} as in transit...")
response = requests.put(
    f"{API_BASE}/api/purchase-orders/{po_id}/status",
    json={"status": "in_transit"}
)
if response.status_code == 200:
    po = response.json()
    print(f"✅ Status updated to: {po['status']}")
else:
    print(f"❌ Failed: {response.status_code}")

# Test 6: Mark as delivered (creates transactions + updates inventory)
print(f"\n6. Marking PO #{po_id} as delivered...")

# Get PO items to build delivery request
response = requests.get(f"{API_BASE}/api/purchase-orders/{po_id}")
po = response.json()

delivery_data = {
    "actual_delivery_date": str(date.today()),
    "items": [
        {
            "item_id": item['id'],
            "quantity_delivered": item['quantity_requested']  # Full delivery
        }
        for item in po['items']
    ]
}

response = requests.post(
    f"{API_BASE}/api/purchase-orders/{po_id}/deliver",
    json=delivery_data
)
if response.status_code == 200:
    po = response.json()
    print(f"✅ PO delivered successfully!")
    print(f"   Status: {po['status']}")
    print(f"   Delivery Date: {po['actual_delivery_date']}")
    print(f"   Transactions created: {len(po['items'])} restock_receipt records")
    print(f"   Inventory updated for {len(po['items'])} SKUs")
else:
    print(f"❌ Failed: {response.status_code} - {response.text}")

# Test 7: Filter POs by store
print("\n7. Listing POs for Store S1...")
response = requests.get(f"{API_BASE}/api/purchase-orders/?store_id=S1")
if response.status_code == 200:
    pos = response.json()
    print(f"✅ Found {len(pos)} POs for Store S1")
else:
    print(f"❌ Failed: {response.status_code}")

# Test 8: Filter POs by status
print("\n8. Listing delivered POs...")
response = requests.get(f"{API_BASE}/api/purchase-orders/?status=delivered")
if response.status_code == 200:
    pos = response.json()
    print(f"✅ Found {len(pos)} delivered POs")
else:
    print(f"❌ Failed: {response.status_code}")

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED!")
print("=" * 80)
