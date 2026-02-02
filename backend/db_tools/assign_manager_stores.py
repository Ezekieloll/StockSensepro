from app.database import SessionLocal
from app.models.user import User

db = SessionLocal()

print("=" * 80)
print("ASSIGNING STORE IDs TO MANAGERS")
print("=" * 80)

# Find managers by name
manager1 = db.query(User).filter(User.name == 'Manager1').first()
manager2 = db.query(User).filter(User.name == 'Manager2').first()
manager3 = db.query(User).filter(User.name == 'Manager3').first()

if manager1:
    manager1.store_id = 'S1'
    print(f"✅ Assigned Manager1 ({manager1.email}) → Store S1")
else:
    print("❌ Manager1 not found")

if manager2:
    manager2.store_id = 'S2'
    print(f"✅ Assigned Manager2 ({manager2.email}) → Store S2")
else:
    print("❌ Manager2 not found")

if manager3:
    manager3.store_id = 'S3'
    print(f"✅ Assigned Manager3 ({manager3.email}) → Store S3")
else:
    print("❌ Manager3 not found")

db.commit()

print("\n" + "=" * 80)
print("VERIFYING ASSIGNMENTS")
print("=" * 80)

# Verify all users
all_users = db.query(User).all()
for user in all_users:
    store_info = f"Store: {user.store_id}" if user.store_id else "No store assigned"
    print(f"{user.name} ({user.role}) - {store_info}")

db.close()

print("\n✅ Store assignments complete!")
