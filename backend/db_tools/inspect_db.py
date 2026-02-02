from app.database import engine
from sqlalchemy import inspect

inspector = inspect(engine)
tables = inspector.get_table_names()

print('TABLES:', tables)
print()

for table in tables:
    print(f'\n{table.upper()}:')
    columns = inspector.get_columns(table)
    for col in columns:
        nullable = "(nullable)" if col["nullable"] else "(NOT NULL)"
        print(f'  - {col["name"]}: {col["type"]} {nullable}')
