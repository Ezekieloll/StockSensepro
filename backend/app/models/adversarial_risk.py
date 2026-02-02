from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from sqlalchemy.sql import func

from app.database import Base


class AdversarialRisk(Base):
    __tablename__ = "adversarial_risk"

    id = Column(Integer, primary_key=True, index=True)

    sku = Column(String, index=True, nullable=False)
    sku_id = Column(String, index=True, nullable=True)  # Added for consistency
    store_id = Column(String, index=True, nullable=False)
    
    # AI Scenario Information
    scenario_name = Column(String, nullable=True)  # e.g., "Holiday Shopping Rush"
    scenario_id = Column(String, index=True, nullable=True)  # e.g., "holiday_rush"
    probability = Column(Float, nullable=True)  # Scenario probability 0-1
    strategies = Column(Text, nullable=True)  # Pipe-separated strategies
    priority_level = Column(String, nullable=True)  # critical, high, medium, low

    baseline_demand = Column(Float, nullable=False)
    worst_case_demand = Column(Float, nullable=False)
    current_inventory = Column(Integer, nullable=True)  # Added for context

    severity = Column(Float, nullable=False)
    days_of_cover = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)

    stockout = Column(Boolean, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
