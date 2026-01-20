from datetime import date
from typing import List, Dict

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.forecast import Forecast


def write_forecasts(
    forecasts: List[Dict],
    model_version: str,
) -> None:
    """
    Save baseline demand forecasts to DB.

    forecasts = [
        {
            "sku": "SKU_001",
            "store_id": "S1",
            "date": date(2026, 1, 10),
            "baseline_demand": 12.5,
        }
    ]
    """

    db: Session = SessionLocal()

    try:
        rows = [
            Forecast(
                sku=f["sku"],
                store_id=f["store_id"],
                date=f["date"],
                baseline_demand=f["baseline_demand"],
                model_version=model_version,
            )
            for f in forecasts
        ]

        db.bulk_save_objects(rows)
        db.commit()

    except Exception as e:
        db.rollback()
        raise e

    finally:
        db.close()
