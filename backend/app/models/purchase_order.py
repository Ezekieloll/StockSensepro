from sqlalchemy import Column, Integer, String, Float, Numeric, DateTime, Date, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class POStatus(str, enum.Enum):
    """Purchase Order status enum"""
    draft = "draft"
    pending = "pending"
    approved = "approved"
    in_transit = "in_transit"
    delivered = "delivered"
    cancelled = "cancelled"


class PurchaseOrder(Base):
    __tablename__ = "purchase_orders"

    id = Column(Integer, primary_key=True, index=True)
    
    # PO identifier
    po_number = Column(String(50), unique=True, index=True, nullable=False)  # PO-2026-001
    
    # Store and destination
    store_id = Column(String(10), nullable=False, index=True)  # S1, S2, S3
    destination = Column(String(50), nullable=False)  # store:S1, store:S2, store:S3
    source = Column(String(50), default="warehouse")  # Always warehouse for now
    
    # Financial summary
    total_items = Column(Integer, nullable=False, default=0)  # Count of line items
    total_quantity = Column(Float, nullable=False, default=0)  # Sum of all quantities
    total_amount = Column(Numeric(12, 2), nullable=True)  # Sum of all line_total
    
    # Status workflow
    status = Column(Enum(POStatus), nullable=False, default=POStatus.pending)
    
    # User tracking
    created_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    approved_by_user_id = Column(Integer, ForeignKey('users.id'), nullable=True)
    
    # Dates
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    approved_date = Column(DateTime(timezone=True), nullable=True)
    expected_delivery_date = Column(Date, nullable=True)
    actual_delivery_date = Column(Date, nullable=True)
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Relationships
    items = relationship("PurchaseOrderItem", back_populates="purchase_order", cascade="all, delete-orphan")
    created_by = relationship("User", foreign_keys=[created_by_user_id])
    approved_by = relationship("User", foreign_keys=[approved_by_user_id])

    def __repr__(self):
        return f"<PurchaseOrder {self.po_number} - {self.status}>"


class PurchaseOrderItem(Base):
    __tablename__ = "purchase_order_items"

    id = Column(Integer, primary_key=True, index=True)
    
    # Link to PO
    purchase_order_id = Column(Integer, ForeignKey('purchase_orders.id', ondelete='CASCADE'), nullable=False)
    
    # Product details
    sku = Column(String(50), nullable=False, index=True)
    product_category = Column(String(10), nullable=True)
    
    # Quantities
    quantity_requested = Column(Float, nullable=False)
    quantity_delivered = Column(Float, nullable=True, default=0)
    
    # Pricing
    unit_price = Column(Numeric(10, 2), nullable=True)
    line_total = Column(Numeric(12, 2), nullable=True)  # quantity_requested * unit_price
    
    # Notes
    notes = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    purchase_order = relationship("PurchaseOrder", back_populates="items")

    def __repr__(self):
        return f"<PurchaseOrderItem PO#{self.purchase_order_id} - {self.sku} x{self.quantity_requested}>"
