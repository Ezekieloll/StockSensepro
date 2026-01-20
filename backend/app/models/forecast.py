from sqlalchemy import Column, Integer, String, Date, Float, DateTime
from sqlalchemy.sql import func

from app.database import Base


class Forecast(Base):
    __tablename__ = "forecasts"

    id = Column(Integer, primary_key=True, index=True)

    sku = Column(String, index=True, nullable=False)
    store_id = Column(String, index=True, nullable=False)

    date = Column(Date, nullable=False)

    baseline_demand = Column(Float, nullable=False)

    model_version = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
