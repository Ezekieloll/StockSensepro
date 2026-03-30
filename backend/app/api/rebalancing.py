from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.inventory import Inventory
from app.models.rebalancing_plan import RebalancingPlan
from app.models.transaction import Transaction
from app.services.audit_utils import write_audit_log
from app.services.auth_utils import get_current_user

router = APIRouter(prefix="/rebalancing", tags=["Rebalancing"])


class RebalancingPlanCreate(BaseModel):
    sku: str = Field(..., min_length=1, max_length=120)
    from_store: str = Field(..., min_length=1, max_length=20)
    to_store: str = Field(..., min_length=1, max_length=20)
    quantity: float = Field(..., gt=0)
    status: str = Field(default="suggested", min_length=1, max_length=20)


VALID_REBALANCING_STATUS = {"suggested", "approved", "executed"}
LOW_STOCK_REMAINING_THRESHOLD = 20.0

@router.get("/")
def get_rebalancing_plans(
    status: str | None = None,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    query = db.query(RebalancingPlan)

    if user.get("role") == "manager" and user.get("store_id"):
        query = query.filter(RebalancingPlan.from_store == user["store_id"])

    if status:
        query = query.filter(RebalancingPlan.status == status)

    return query.order_by(RebalancingPlan.created_at.desc()).all()


@router.post("/")
def create_rebalancing_plan(
    payload: RebalancingPlanCreate,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    if payload.from_store == payload.to_store:
        raise HTTPException(status_code=400, detail="from_store and to_store must be different")

    status_value = payload.status.strip().lower() or "suggested"
    if status_value not in VALID_REBALANCING_STATUS:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status_value}")

    if user.get("role") == "manager" and user.get("store_id") and payload.from_store.strip() != user["store_id"]:
        raise HTTPException(status_code=403, detail="Managers can only create plans from their assigned store")

    plan = RebalancingPlan(
        sku=payload.sku.strip(),
        from_store=payload.from_store.strip(),
        to_store=payload.to_store.strip(),
        quantity=payload.quantity,
        status=status_value,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)

    write_audit_log(
        db,
        action="REBALANCING_PLAN_CREATED",
        user_id=int(user.get("sub")) if user.get("sub") else None,
        entity=f"rebalancing_plan:{plan.id}",
        details={
            "sku": plan.sku,
            "from_store": plan.from_store,
            "to_store": plan.to_store,
            "quantity": plan.quantity,
            "status": plan.status,
        },
    )
    db.commit()

    return plan


@router.post("/{plan_id}/execute")
def execute_rebalancing_plan(
    plan_id: int,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
    request: Request = None,
):
    plan = db.query(RebalancingPlan).filter(RebalancingPlan.id == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Rebalancing plan not found")

    if plan.status == "executed":
        raise HTTPException(status_code=400, detail="Plan already executed")

    if user.get("role") == "manager" and user.get("store_id") and plan.from_store != user["store_id"]:
        raise HTTPException(status_code=403, detail="Managers can only execute plans from their assigned store")

    source_inventory = db.query(Inventory).filter(
        Inventory.sku == plan.sku,
        Inventory.store_id == plan.from_store,
    ).first()

    if not source_inventory:
        raise HTTPException(status_code=400, detail="Source inventory record not found")

    if source_inventory.quantity < plan.quantity:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient source inventory: {source_inventory.quantity} available, {plan.quantity} requested",
        )

    destination_inventory = db.query(Inventory).filter(
        Inventory.sku == plan.sku,
        Inventory.store_id == plan.to_store,
    ).first()

    source_before = float(source_inventory.quantity)
    source_after = source_before - float(plan.quantity)

    dest_before = float(destination_inventory.quantity) if destination_inventory else 0.0
    dest_after = dest_before + float(plan.quantity)

    source_inventory.quantity = source_after
    if destination_inventory:
        destination_inventory.quantity = dest_after
    else:
        destination_inventory = Inventory(
            sku=plan.sku,
            store_id=plan.to_store,
            quantity=dest_after,
        )
        db.add(destination_inventory)

    transfer_timestamp = datetime.now()
    transfer_date = date.today()

    source_tx = Transaction(
        timestamp=transfer_timestamp,
        date=transfer_date,
        store_id=plan.from_store,
        product_id=plan.sku,
        product_category=None,
        event_type="transfer_out",
        quantity=float(plan.quantity),
        on_hand_before=int(source_before),
        on_hand_after=int(source_after),
        source=f"store:{plan.from_store}",
        destination=f"store:{plan.to_store}",
        price=None,
        is_simulated=0,
    )
    db.add(source_tx)

    dest_tx = Transaction(
        timestamp=transfer_timestamp,
        date=transfer_date,
        store_id=plan.to_store,
        product_id=plan.sku,
        product_category=None,
        event_type="transfer_in",
        quantity=float(plan.quantity),
        on_hand_before=int(dest_before),
        on_hand_after=int(dest_after),
        source=f"store:{plan.from_store}",
        destination=f"store:{plan.to_store}",
        price=None,
        is_simulated=0,
    )
    db.add(dest_tx)

    plan.status = "executed"

    warning: str | None = None
    if source_after <= LOW_STOCK_REMAINING_THRESHOLD:
        warning = (
            f"Warning: source store {plan.from_store} will have low remaining stock "
            f"for {plan.sku} after transfer ({source_after:.0f} units)."
        )

    db.commit()
    db.refresh(plan)

    write_audit_log(
        db,
        action="REBALANCING_PLAN_EXECUTED",
        user_id=int(user.get("sub")) if user.get("sub") else None,
        entity=f"rebalancing_plan:{plan.id}",
        details={
            "sku": plan.sku,
            "from_store": plan.from_store,
            "to_store": plan.to_store,
            "quantity": plan.quantity,
            "source_before": source_before,
            "source_after": source_after,
            "dest_before": dest_before,
            "dest_after": dest_after,
        },
        ip_address=request.client.host if request and request.client else None,
    )
    db.commit()

    return {
        "message": "Rebalancing plan executed",
        "warning": warning,
        "plan": plan,
        "inventory": {
            "source": {
                "store_id": plan.from_store,
                "sku": plan.sku,
                "before": source_before,
                "after": source_after,
            },
            "destination": {
                "store_id": plan.to_store,
                "sku": plan.sku,
                "before": dest_before,
                "after": dest_after,
            },
        },
    }
