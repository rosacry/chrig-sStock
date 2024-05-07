from fastapi import APIRouter, HTTPException, status
from ..schemas import user_schema

router = APIRouter()

@router.post("/register", response_model=user_schema.UserResponse)
async def register_user(user: user_schema.UserCreate):
    # Here you would typically insert the user into the database
    # For now, we'll simulate success
    return {"email": user.email, "full_name": user.full_name}