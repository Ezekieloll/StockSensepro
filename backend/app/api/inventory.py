from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.inventory import Inventory
from app.services.auth_utils import get_current_user

router = APIRouter(prefix="/inventory", tags=["Inventory"])

@router.get("/")
def get_inventory(
    sku: str | None = None,
    store_id: str | None = None,
    db: Session = Depends(get_db),
    user: dict = Depends(get_current_user),
):
    # Managers are restricted to their assigned store.
    if user.get("role") == "manager" and user.get("store_id"):
        store_id = user["store_id"]

    query = db.query(Inventory)

    if sku:
        query = query.filter(Inventory.sku == sku)
    if store_id:
        query = query.filter(Inventory.store_id == store_id)

    return query.all()
