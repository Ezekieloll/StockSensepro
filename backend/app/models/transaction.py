"""
Transaction model for storing historical and live transaction data.
Column names match the CSV files exactly.
"""

from sqlalchemy import Column, Integer, Float, String, Date, DateTime, Index
from sqlalchemy.sql import func

from app.database import Base


class Transaction(Base):
    """
    Stores all transaction events (sales, returns, restocks).
    Column names match CSV exactly:
    timestamp, date, store_id, product_id, product_category, event_type,
    quantity, on_hand_before, on_hand_after, source, destination, price,
    holiday_flag, weather
    """
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Core transaction data (matching CSV columns exactly)
    timestamp = Column(DateTime, nullable=True)
    date = Column(Date, nullable=False, index=True)
    store_id = Column(String(10), nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    product_category = Column(String(10), nullable=True)
    event_type = Column(String(20), nullable=False)  # sale, return, restock
    quantity = Column(Float, nullable=False, default=1)
    on_hand_before = Column(Integer, nullable=True)
    on_hand_after = Column(Integer, nullable=True)
    source = Column(String(50), nullable=True)
    destination = Column(String(50), nullable=True)
    price = Column(Float, nullable=True)
    holiday_flag = Column(Integer, nullable=True)  # 0 or 1
    weather = Column(String(50), nullable=True)
    
    # Metadata (not in CSV)
    is_simulated = Column(Integer, default=0)  # 0 = real data, 1 = simulated
    created_at = Column(DateTime, server_default=func.now())
    
    # Compound indexes for efficient queries
    __table_args__ = (
        Index('ix_transactions_store_product', 'store_id', 'product_id'),
        Index('ix_transactions_date_product', 'date', 'product_id'),
        Index('ix_transactions_store_date', 'store_id', 'date'),
    )
    
    def __repr__(self):
        return f"<Transaction {self.store_id}/{self.product_id}: {self.event_type} x{self.quantity}>"


class DailyDemand(Base):
    """
    Aggregated daily demand per product per store.
    Pre-computed for faster chart queries.
    """
    __tablename__ = "daily_demand"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    date = Column(Date, nullable=False, index=True)
    store_id = Column(String(10), nullable=False, index=True)
    product_id = Column(String(50), nullable=False, index=True)
    product_category = Column(String(10), nullable=True)
    
    # Aggregated values
    total_quantity = Column(Float, nullable=False, default=0)
    transaction_count = Column(Integer, nullable=False, default=0)
    total_revenue = Column(Float, nullable=True)
    avg_price = Column(Float, nullable=True)
    
    # For tracking data updates
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index('ix_daily_demand_unique', 'date', 'store_id', 'product_id', unique=True),
    )
    
    def __repr__(self):
        return f"<DailyDemand {self.date} {self.store_id}/{self.product_id}: {self.total_quantity}>"
