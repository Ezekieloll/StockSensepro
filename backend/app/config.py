import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY") or os.getenv("JWT_SECRET")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ENV = os.getenv("ENV", "development").strip().lower()

if not SECRET_KEY and ENV != "production":
    # Dev fallback to avoid auth 500s when env vars are not configured yet.
    SECRET_KEY = "stocksense-dev-secret-change-me"

DEFAULT_ALLOWED_ORIGINS = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000"
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS).split(",")
    if origin.strip()
]

ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX", "").strip() or None

if ENV != "production" and ALLOWED_ORIGIN_REGEX is None:
    # Allow common private-network dev origins (e.g. opening Next.js from a phone/LAN IP).
    ALLOWED_ORIGIN_REGEX = (
        r"^https?://"
        r"(localhost|127\.0\.0\.1|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3})"
        r"(:\d+)?$"
    )

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")
