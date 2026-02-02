"""
Staging tables for CSV uploads before validation and approval.
Allows admin to review data before committing to master DB.
"""

from sqlalchemy import Column, Integer, Float, String, Date, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

from app.database import Base


class StagingUpload(Base):
    """
    Tracks CSV upload sessions and their overall status.
    """
    __tablename__ = "staging_uploads"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Upload metadata
    filename = Column(String(255), nullable=False)
    uploaded_by = Column(String(100), nullable=False)  # username/email
    uploaded_at = Column(DateTime, server_default=func.now())
    
    # Status tracking
    status = Column(String(20), nullable=False, default='pending')  # pending, approved, rejected, error
    row_count = Column(Integer, default=0)
    valid_rows = Column(Integer, default=0)
    invalid_rows = Column(Integer, default=0)
    
    # Validation results
    error_message = Column(Text, nullable=True)
    validation_summary = Column(Text, nullable=True)  # JSON string with details
    
    # Date range of data
    min_date = Column(Date, nullable=True)
    max_date = Column(Date, nullable=True)
    
    # Metadata
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    transactions = relationship("StagingTransaction", back_populates="upload", cascade="all, delete-orphan")


class StagingTransaction(Base):
    """
    Temporary storage for uploaded transaction data before approval.
    Mirrors Transaction model structure.
    """
    __tablename__ = "staging_transactions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    upload_id = Column(Integer, ForeignKey('staging_uploads.id', ondelete='CASCADE'), nullable=False, index=True)
    
    # Core transaction data (matching CSV columns)
    timestamp = Column(DateTime, nullable=True)
    date = Column(Date, nullable=False)
    store_id = Column(String(10), nullable=False)
    product_id = Column(String(50), nullable=False)
    product_category = Column(String(10), nullable=True)
    event_type = Column(String(20), nullable=False)
    quantity = Column(Float, nullable=False)
    on_hand_before = Column(Integer, nullable=True)
    on_hand_after = Column(Integer, nullable=True)
    source = Column(String(50), nullable=True)
    destination = Column(String(50), nullable=True)
    price = Column(Float, nullable=True)
    holiday_flag = Column(Integer, nullable=True)
    weather = Column(String(50), nullable=True)
    
    # Validation status for this row
    is_valid = Column(Integer, default=1)  # 1 = valid, 0 = invalid
    validation_error = Column(String(255), nullable=True)
    
    # Relationships
    upload = relationship("StagingUpload", back_populates="transactions")
