from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func

from app.database import Base


class RebalancingPlan(Base):
    __tablename__ = "rebalancing_plan"

    id = Column(Integer, primary_key=True, index=True)

    sku = Column(String, index=True, nullable=False)

    from_store = Column(String, nullable=False)
    to_store = Column(String, nullable=False)

    quantity = Column(Float, nullable=False)

    status = Column(String, nullable=False, default="suggested")
    # suggested | approved | executed

    created_at = Column(DateTime(timezone=True), server_default=func.now())
