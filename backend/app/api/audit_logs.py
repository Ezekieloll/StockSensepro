"""
Audit logs API for admin activity timeline.
"""
import json

from fastapi import APIRouter, Depends
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.audit_log import AuditLog
from app.models.user import User

router = APIRouter(prefix="/api/audit-logs", tags=["audit-logs"])


@router.get("/")
def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)):
    """Get recent audit logs ordered newest-first."""
    safe_limit = max(1, min(limit, 200))

    try:
        logs = (
            db.query(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(safe_limit)
            .all()
        )
    except SQLAlchemyError:
        # Keep dashboard functional when audit table migration has not run yet.
        return []

    user_ids = sorted({log.user_id for log in logs if log.user_id is not None})
    users = (
        db.query(User)
        .filter(User.id.in_(user_ids))
        .all()
        if user_ids
        else []
    )
    user_lookup = {user.id: user for user in users}

    response = []
    for log in logs:
        parsed_details = None
        if log.details:
            try:
                parsed_details = json.loads(log.details)
            except json.JSONDecodeError:
                parsed_details = {"raw": log.details}

        actor = user_lookup.get(log.user_id) if log.user_id is not None else None

        response.append(
            {
                "id": log.id,
                "action": log.action,
                "entity": log.entity,
                "details": parsed_details,
                "ip_address": log.ip_address,
                "created_at": log.created_at.isoformat() if log.created_at else None,
                "user": {
                    "id": actor.id,
                    "name": actor.name,
                    "email": actor.email,
                    "role": actor.role,
                }
                if actor
                else None,
            }
        )

    return response
