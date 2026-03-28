from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from typing import Generator
from app.config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

Base = declarative_base()



def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
