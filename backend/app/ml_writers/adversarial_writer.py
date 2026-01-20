from typing import List, Dict

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models.adversarial_risk import AdversarialRisk


def write_adversarial_risk(
    risks: List[Dict],
) -> None:
    """
    Save adversarial risk analysis to DB.

    risks = [
        {
            "sku": "SKU_001",
            "store_id": "S1",
            "baseline_demand": 10,
            "worst_case_demand": 20,
            "severity": 1.0,
            "days_of_cover": 1.2,
            "risk_score": 0.95,
            "stockout": True,
        }
    ]
    """

    db: Session = SessionLocal()

    try:
        rows = [
            AdversarialRisk(
                sku=r["sku"],
                store_id=r["store_id"],
                baseline_demand=r["baseline_demand"],
                worst_case_demand=r["worst_case_demand"],
                severity=r["severity"],
                days_of_cover=r["days_of_cover"],
                risk_score=r["risk_score"],
                stockout=r["stockout"],
            )
            for r in risks
        ]

        db.bulk_save_objects(rows)
        db.commit()

    except Exception as e:
        db.rollback()
        raise e

    finally:
        db.close()
