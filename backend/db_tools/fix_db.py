import sqlalchemy
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:Xuv500w8@localhost:5432/stocksense"
engine = create_engine(DATABASE_URL)

with engine.connect() as connection:
    # 1. Drop users table if it exists
    print("Dropping users table...")
    try:
        connection.execute(text("DROP TABLE IF EXISTS users CASCADE;"))
        print("Dropped users table.")
    except Exception as e:
        print(f"Error dropping users: {e}")

    # 2. Update alembic version to the one before users creation
    # The migration that creates users is 47430731dfc2, its down_revision is 7875da9ba487
    target_revision = '7875da9ba487'
    print(f"Setting alembic version to {target_revision}...")
    try:
        # Check if row exists
        result = connection.execute(text("SELECT * FROM alembic_version"))
        if result.rowcount > 0:
            connection.execute(text(f"UPDATE alembic_version SET version_num = '{target_revision}';"))
        else:
             connection.execute(text(f"INSERT INTO alembic_version (version_num) VALUES ('{target_revision}');"))
        connection.commit()
        print("Alembic version updated.")
    except Exception as e:
        print(f"Error updating alembic version: {e}")
