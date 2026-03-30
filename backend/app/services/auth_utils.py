import bcrypt
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

from app.config import SECRET_KEY, ALGORITHM


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def _require_secret_key() -> str:
    """Return SECRET_KEY or fail fast when auth is used without configuration."""
    if not SECRET_KEY:
        raise RuntimeError("SECRET_KEY is not set")
    return SECRET_KEY


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = password.encode('utf-8')
    hashed_bytes = hashed.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def create_access_token(
    user_id: int,
    role: str,
    store_id: str | None,
    expires_hours: int = 8,
) -> str:
    """Create a signed JWT token for API authentication."""
    secret_key = _require_secret_key()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
    payload = {
        "sub": str(user_id),
        "role": role,
        "store_id": store_id,
        "exp": expires_at,
    }
    return jwt.encode(payload, secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Decode and validate JWT token payload."""
    secret_key = _require_secret_key()
    try:
        return jwt.decode(token, secret_key, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        ) from exc


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency for endpoints that need current user claims."""
    return decode_access_token(token)


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency that restricts access to admin users."""
    if user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return user
