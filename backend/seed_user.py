from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import sys
import os

# Ensure app can be imported
sys.path.append(os.getcwd())

from app.models.user import User
from app.services.auth_utils import hash_password

DATABASE_URL = "postgresql://postgres:Xuv500w8@localhost:5432/stocksense"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
db = SessionLocal()

print("Seeding users...")
email = "manager@stocksense.com"
# Need to handle potential rollback if error occurs
try:
    existing = db.query(User).filter(User.email == email).first()
    if not existing:
        user = User(
            name="Manager", 
            email=email,
            password_hash=hash_password("password"),
            role="manager"
        )
        db.add(user)
        db.commit()
        print(f"Created user {email} with password 'password'")
    else:
        print(f"User {email} already exists")
except Exception as e:
    print(f"Error seeding user: {e}")
    db.rollback()
finally:
    db.close()
