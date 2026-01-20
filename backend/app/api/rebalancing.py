from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.rebalancing_plan import RebalancingPlan

router = APIRouter(prefix="/rebalancing", tags=["Rebalancing"])

@router.get("/")
def get_rebalancing_plans(
    status: str | None = None,
    db: Session = Depends(get_db),
):
    query = db.query(RebalancingPlan)

    if status:
        query = query.filter(RebalancingPlan.status == status)

    return query.order_by(RebalancingPlan.created_at.desc()).all()
