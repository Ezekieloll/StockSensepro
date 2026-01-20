from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.adversarial_risk import AdversarialRisk

router = APIRouter(prefix="/adversarial", tags=["Adversarial"])

@router.get("/")
def get_adversarial_risk(
    sku: str | None = None,
    store_id: str | None = None,
    high_risk_only: bool = False,
    db: Session = Depends(get_db),
):
    query = db.query(AdversarialRisk)

    if sku:
        query = query.filter(AdversarialRisk.sku == sku)
    if store_id:
        query = query.filter(AdversarialRisk.store_id == store_id)
    if high_risk_only:
        query = query.filter(AdversarialRisk.stockout == True)

    return query.order_by(AdversarialRisk.risk_score.desc()).all()
