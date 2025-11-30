from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from typing import List
import json
import os

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

USERS_FILE = "users.json"

# Helper function to load users
def load_users():
    if not os.path.exists(USERS_FILE):
        return []
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# Helper function to save users
def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=4)

# Example route: get all users
@router.get("/", tags=["users"])
async def get_all_users():
    users = load_users()
    # don't return passwords
    for u in users:
        u.pop("password", None)
    return users

# Example route: get current user info
@router.get("/me", tags=["users"])
async def get_current_user(token: str = Depends(oauth2_scheme)):
    users = load_users()
    # for now just return first user as placeholder
    if not users:
        raise HTTPException(status_code=404, detail="No users found")
    user = users[0]
    user.pop("password", None)
    return user


