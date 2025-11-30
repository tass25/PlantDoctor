from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator
from datetime import datetime
from utils.json_handler import read_json, write_json, find_json_item
from utils.hashing import hash_password, verify_password
from utils.jwt_handler import create_access_token
from config import settings
import uuid

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Pydantic models
class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3)
    password: str = Field(..., min_length=6)
    confirm_password: str
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (underscores and hyphens allowed)')
        return v
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

@router.post("/register", response_model=AuthResponse)
async def register(data: RegisterRequest):
    """
    Register a new user account.
    """
    users = read_json(settings.USERS_FILE)
    
    # Check if username already exists
    if find_json_item(settings.USERS_FILE, lambda u: u.get("username") == data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Create new user
    user_id = str(uuid.uuid4())
    new_user = {
        "id": user_id,
        "username": data.username,
        "password": hash_password(data.password),
        "role": "user",
        "created_at": datetime.utcnow().isoformat(),
        "stats": {
            "total_points": 0,
            "total_scans": 0,
            "plants_saved": 0,
            "diseases_found": 0,
            "healthy_plants": 0,
            "perfect_scans": 0,
            "average_accuracy": 0,
            "badges_earned": 0
        }
    }
    
    users.append(new_user)
    write_json(settings.USERS_FILE, users)
    
    # Create JWT token
    token = create_access_token({
        "sub": data.username,
        "role": "user"
    })
    
    # Return user without password
    user_data = {k: v for k, v in new_user.items() if k != "password"}
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }

@router.post("/login", response_model=AuthResponse)
async def login(data: LoginRequest):
    """
    Login with username and password.
    """
    user = find_json_item(settings.USERS_FILE, lambda u: u.get("username") == data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password
    if not verify_password(data.password, user.get("password")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create JWT token
    token = create_access_token({
        "sub": user["username"],
        "role": user.get("role", "user")
    })
    
    # Return user without password
    user_data = {k: v for k, v in user.items() if k != "password"}
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": user_data
    }