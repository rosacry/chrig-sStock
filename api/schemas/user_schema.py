from pydantic import BaseModel

class UserCreate(BaseModel):
    email: str
    full_name: str

class UserResponse(BaseModel):
    email: str
    full_name: str