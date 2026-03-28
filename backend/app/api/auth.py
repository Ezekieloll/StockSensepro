import logging

from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.database import get_db
from app.models.user import User
from app.services.auth_utils import hash_password, verify_password, create_access_token
from app.services.audit_utils import write_audit_log

router = APIRouter(prefix="/api/auth", tags=["auth"])
logger = logging.getLogger(__name__)


def _safe_write_audit_log(
    db: Session,
    action: str,
    user_id: int | None = None,
    entity: str | None = None,
    details: dict | None = None,
    ip_address: str | None = None,
) -> None:
    """Best-effort audit logging; never break auth flows if audit table is unavailable."""
    try:
        write_audit_log(
            db,
            action=action,
            user_id=user_id,
            entity=entity,
            details=details,
            ip_address=ip_address,
        )
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        logger.warning("Audit logging skipped for action %s: %s", action, exc)


class SignupRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    identifier: str = Field(..., min_length=2, max_length=200)
    password: str = Field(..., min_length=8, max_length=128)


@router.post("/signup")
def signup(data: SignupRequest, db: Session = Depends(get_db), request: Request = None):
    if db.query(User).filter(User.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        name=data.name,
        email=str(data.email),
        password_hash=hash_password(data.password),
        role="analyst",
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    _safe_write_audit_log(
        db,
        action="AUTH_SIGNUP_SUCCESS",
        user_id=user.id,
        entity=f"user:{user.id}",
        details={"email": user.email},
        ip_address=request.client.host if request and request.client else None,
    )

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
    }


@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db), request: Request = None):
    user = (
        db.query(User)
        .filter(
            (User.email == data.identifier) |
            (User.name == data.identifier)
        )
        .first()
    )

    if not user or not verify_password(data.password, user.password_hash):
        _safe_write_audit_log(
            db,
            action="AUTH_LOGIN_FAILED",
            entity="auth:login",
            details={"identifier": data.identifier},
            ip_address=request.client.host if request and request.client else None,
        )
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id, user.role, user.store_id)

    _safe_write_audit_log(
        db,
        action="AUTH_LOGIN_SUCCESS",
        user_id=user.id,
        entity="auth:login",
        details={"role": user.role},
        ip_address=request.client.host if request and request.client else None,
    )

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "store_id": user.store_id,
        },
    }
