from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)

    role = Column(String, nullable=False)  # admin | manager | analyst
    store_id = Column(String, nullable=True)  # S1, S2, S3 (only for managers)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
