from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, date
from pydantic import BaseModel

from app.database import get_db
from app.models.purchase_order import PurchaseOrder, PurchaseOrderItem, POStatus
from app.models.user import User
from app.models.transaction import Transaction
from app.models.inventory import Inventory

router = APIRouter(prefix="/api/purchase-orders", tags=["purchase-orders"])


# Pydantic schemas
class PurchaseOrderItemCreate(BaseModel):
    sku: str
    product_category: Optional[str] = None
    quantity_requested: float
    unit_price: Optional[float] = None
    notes: Optional[str] = None


class PurchaseOrderItemResponse(BaseModel):
    id: int
    sku: str
    product_category: Optional[str]
    quantity_requested: float
    quantity_delivered: float
    unit_price: Optional[float]
    line_total: Optional[float]
    notes: Optional[str]

    class Config:
        from_attributes = True


class PurchaseOrderCreate(BaseModel):
    store_id: str
    items: List[PurchaseOrderItemCreate]
    expected_delivery_date: Optional[date] = None
    notes: Optional[str] = None
    created_by_user_id: int  # In real app, get from auth token


class PurchaseOrderResponse(BaseModel):
    id: int
    po_number: str
    store_id: str
    destination: str
    source: str
    total_items: int
    total_quantity: float
    total_amount: Optional[float]
    status: str
    created_by_user_id: int
    approved_by_user_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    approved_date: Optional[datetime]
    expected_delivery_date: Optional[date]
    actual_delivery_date: Optional[date]
    notes: Optional[str]
    items: List[PurchaseOrderItemResponse] = []

    class Config:
        from_attributes = True


class PurchaseOrderStatusUpdate(BaseModel):
    status: str  # pending, approved, in_transit, delivered, cancelled


class DeliveryItem(BaseModel):
    item_id: int
    quantity_delivered: float


class DeliveryRequest(BaseModel):
    actual_delivery_date: date
    items: List[DeliveryItem]


# Helper function to generate PO number
def generate_po_number(db: Session) -> str:
    """Generate unique PO number like PO-2026-001"""
    year = datetime.now().year
    
    # Get the latest PO number for this year
    latest_po = db.query(PurchaseOrder).filter(
        PurchaseOrder.po_number.like(f"PO-{year}-%")
    ).order_by(PurchaseOrder.id.desc()).first()
    
    if latest_po:
        # Extract number and increment
        last_num = int(latest_po.po_number.split('-')[-1])
        new_num = last_num + 1
    else:
        new_num = 1
    
    return f"PO-{year}-{new_num:03d}"


@router.get("/", response_model=List[PurchaseOrderResponse])
def list_purchase_orders(
    store_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """List all purchase orders with optional filters"""
    query = db.query(PurchaseOrder)
    
    if store_id:
        query = query.filter(PurchaseOrder.store_id == store_id)
    
    if status:
        query = query.filter(PurchaseOrder.status == status)
    
    pos = query.order_by(PurchaseOrder.created_at.desc()).limit(limit).all()
    return pos


@router.get("/{po_id}", response_model=PurchaseOrderResponse)
def get_purchase_order(po_id: int, db: Session = Depends(get_db)):
    """Get single PO with all items"""
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    
    if not po:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    
    return po


@router.post("/", response_model=PurchaseOrderResponse, status_code=status.HTTP_201_CREATED)
def create_purchase_order(po_data: PurchaseOrderCreate, db: Session = Depends(get_db)):
    """Create new purchase order"""
    
    # Generate PO number
    po_number = generate_po_number(db)
    
    # Calculate totals
    total_items = len(po_data.items)
    total_quantity = sum(item.quantity_requested for item in po_data.items)
    total_amount = sum(
        (item.quantity_requested * item.unit_price) 
        for item in po_data.items 
        if item.unit_price is not None
    )
    
    # Create PO
    new_po = PurchaseOrder(
        po_number=po_number,
        store_id=po_data.store_id,
        destination=f"store:{po_data.store_id}",
        source="warehouse",
        total_items=total_items,
        total_quantity=total_quantity,
        total_amount=total_amount if total_amount > 0 else None,
        status=POStatus.pending,
        created_by_user_id=po_data.created_by_user_id,
        expected_delivery_date=po_data.expected_delivery_date,
        notes=po_data.notes
    )
    
    db.add(new_po)
    db.flush()  # Get the PO id
    
    # Create PO items
    for item_data in po_data.items:
        line_total = None
        if item_data.unit_price is not None:
            line_total = item_data.quantity_requested * item_data.unit_price
        
        po_item = PurchaseOrderItem(
            purchase_order_id=new_po.id,
            sku=item_data.sku,
            product_category=item_data.product_category,
            quantity_requested=item_data.quantity_requested,
            quantity_delivered=0,
            unit_price=item_data.unit_price,
            line_total=line_total,
            notes=item_data.notes
        )
        db.add(po_item)
    
    db.commit()
    db.refresh(new_po)
    
    return new_po


@router.put("/{po_id}/status", response_model=PurchaseOrderResponse)
def update_po_status(
    po_id: int, 
    status_update: PurchaseOrderStatusUpdate, 
    db: Session = Depends(get_db)
):
    """Update PO status (pending → approved → in_transit → delivered)"""
    
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    
    if not po:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    
    # Validate status transition
    valid_statuses = ['pending', 'approved', 'in_transit', 'delivered', 'cancelled']
    if status_update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status_update.status}")
    
    po.status = POStatus(status_update.status)
    po.updated_at = datetime.now()
    
    # Set approved_date when status changes to approved
    if status_update.status == 'approved' and not po.approved_date:
        po.approved_date = datetime.now()
    
    db.commit()
    db.refresh(po)
    
    return po


@router.post("/{po_id}/deliver", response_model=PurchaseOrderResponse)
def mark_po_delivered(
    po_id: int,
    delivery: DeliveryRequest,
    db: Session = Depends(get_db)
):
    """Mark PO as delivered and create restock_receipt transactions"""
    
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    
    if not po:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    
    if po.status == POStatus.delivered:
        raise HTTPException(status_code=400, detail="PO already delivered")
    
    # Update PO status
    po.status = POStatus.delivered
    po.actual_delivery_date = delivery.actual_delivery_date
    po.updated_at = datetime.now()
    
    # Process each delivered item
    for delivery_item in delivery.items:
        po_item = db.query(PurchaseOrderItem).filter(
            PurchaseOrderItem.id == delivery_item.item_id
        ).first()
        
        if not po_item or po_item.purchase_order_id != po_id:
            continue
        
        # Update delivered quantity
        po_item.quantity_delivered = delivery_item.quantity_delivered
        
        # Get current inventory
        inventory = db.query(Inventory).filter(
            Inventory.sku == po_item.sku,
            Inventory.store_id == po.store_id
        ).first()
        
        on_hand_before = inventory.quantity if inventory else 0
        on_hand_after = on_hand_before + delivery_item.quantity_delivered
        
        # Create restock_receipt transaction
        restock_tx = Transaction(
            date=delivery.actual_delivery_date,
            timestamp=datetime.now(),
            store_id=po.store_id,
            product_id=po_item.sku,
            product_category=po_item.product_category,
            event_type='restock_receipt',
            quantity=delivery_item.quantity_delivered,
            source='warehouse',
            destination=f'store:{po.store_id}',
            price=float(po_item.unit_price) if po_item.unit_price else None,
            on_hand_before=int(on_hand_before),
            on_hand_after=int(on_hand_after),
            is_simulated=0
        )
        db.add(restock_tx)
        
        # Update inventory
        if inventory:
            inventory.quantity = on_hand_after
        else:
            # Create new inventory record if doesn't exist
            new_inventory = Inventory(
                sku=po_item.sku,
                store_id=po.store_id,
                quantity=delivery_item.quantity_delivered
            )
            db.add(new_inventory)
    
    db.commit()
    db.refresh(po)
    
    return po


@router.delete("/{po_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_purchase_order(po_id: int, db: Session = Depends(get_db)):
    """Delete PO (only if status is draft or pending)"""
    
    po = db.query(PurchaseOrder).filter(PurchaseOrder.id == po_id).first()
    
    if not po:
        raise HTTPException(status_code=404, detail="Purchase order not found")
    
    if po.status not in [POStatus.draft, POStatus.pending]:
        raise HTTPException(
            status_code=400, 
            detail="Cannot delete PO that is already approved or delivered"
        )
    
    db.delete(po)
    db.commit()
    
    return None
