"""
User management API for admin operations.
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional

from app.database import get_db
from app.models.user import User
from app.services.auth_utils import hash_password

router = APIRouter(prefix="/api/users", tags=["users"])


class UserCreate(BaseModel):
    name: str
    email: str
    password: str
    role: str  # admin, manager, analyst


class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    password: Optional[str] = None


@router.get("/")
def get_all_users(db: Session = Depends(get_db)):
    """Get all users (admin only)"""
    users = db.query(User).order_by(User.id).all()
    return [
        {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role,
            "created_at": user.created_at.isoformat() if user.created_at else None
        }
        for user in users
    ]


@router.post("/")
def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create a new user (admin only)"""
    # Check if email already exists
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Validate role
    valid_roles = ["admin", "manager", "analyst"]
    if user_data.role not in valid_roles:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
        )
    
    # Create user
    user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        role=user_data.role,
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }


@router.put("/{user_id}")
def update_user(user_id: int, user_data: UserUpdate, db: Session = Depends(get_db)):
    """Update user details (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update fields
    if user_data.name is not None:
        user.name = user_data.name
    if user_data.email is not None:
        # Check if new email is already taken by another user
        existing = db.query(User).filter(
            User.email == user_data.email,
            User.id != user_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already in use")
        user.email = user_data.email
    if user_data.role is not None:
        valid_roles = ["admin", "manager", "analyst"]
        if user_data.role not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role. Must be one of: {', '.join(valid_roles)}"
            )
        user.role = user_data.role
    if user_data.password is not None:
        user.password_hash = hash_password(user_data.password)
    
    db.commit()
    db.refresh(user)
    
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }


@router.delete("/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user (admin only)"""
    user = db.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Prevent deleting the last admin
    if user.role == "admin":
        admin_count = db.query(User).filter(User.role == "admin").count()
        if admin_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot delete the last admin user"
            )
    
    db.delete(user)
    db.commit()
    
    return {"message": f"User {user.email} deleted successfully"}
