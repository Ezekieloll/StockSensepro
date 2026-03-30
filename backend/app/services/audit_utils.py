import json

from sqlalchemy.orm import Session

from app.models.audit_log import AuditLog


def write_audit_log(
    db: Session,
    action: str,
    user_id: int | None = None,
    entity: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """Persist an audit event in the current DB transaction."""
    entry = AuditLog(
        user_id=user_id,
        action=action,
        entity=entity,
        details=json.dumps(details) if details else None,
        ip_address=ip_address,
    )
    db.add(entry)
