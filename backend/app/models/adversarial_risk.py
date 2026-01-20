from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from sqlalchemy.sql import func

from app.database import Base


class AdversarialRisk(Base):
    __tablename__ = "adversarial_risk"

    id = Column(Integer, primary_key=True, index=True)

    sku = Column(String, index=True, nullable=False)
    store_id = Column(String, index=True, nullable=False)

    baseline_demand = Column(Float, nullable=False)
    worst_case_demand = Column(Float, nullable=False)

    severity = Column(Float, nullable=False)
    days_of_cover = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)

    stockout = Column(Boolean, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
