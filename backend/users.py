from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from utils.json_handler import find_json_item, find_all_json_items
from utils.jwt_handler import get_current_user
from config import settings

router = APIRouter(prefix="/users", tags=["Users"])

class UserProfile(BaseModel):
    id: str
    username: str
    role: str
    created_at: str
    stats: dict

class DashboardStats(BaseModel):
    user: UserProfile
    recent_analyses: list
    badges: list
    ranking: dict

@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user's profile.
    """
    user = find_json_item(
        settings.USERS_FILE, 
        lambda u: u.get("username") == current_user["username"]
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Remove password from response
    user_data = {k: v for k, v in user.items() if k != "password"}
    return user_data

@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(current_user: dict = Depends(get_current_user)):
    """
    Get dashboard statistics for current user.
    """
    # Get user
    user = find_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == current_user["username"]
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Get user's history
    user_history = find_all_json_items(
        settings.HISTORY_FILE,
        lambda h: h.get("username") == current_user["username"]
    )
    
    # Sort by timestamp descending and get recent 5
    user_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    recent_analyses = user_history[:5]
    
    # Get user's badges
    all_badges = find_all_json_items(settings.BADGES_FILE)
    user_badges = [b for b in all_badges if b.get("username") == current_user["username"]]
    
    # Calculate ranking
    all_users = find_all_json_items(settings.USERS_FILE)
    all_users.sort(key=lambda x: x.get("stats", {}).get("total_points", 0), reverse=True)
    
    user_rank = 0
    for idx, u in enumerate(all_users, 1):
        if u.get("username") == current_user["username"]:
            user_rank = idx
            break
    
    ranking = {
        "rank": user_rank,
        "total_users": len(all_users),
        "percentile": round((1 - (user_rank / len(all_users))) * 100, 1) if all_users else 0
    }
    
    # Remove password from user data
    user_data = {k: v for k, v in user.items() if k != "password"}
    
    return {
        "user": user_data,
        "recent_analyses": recent_analyses,
        "badges": user_badges,
        "ranking": ranking
    }

@router.get("/profile/{username}")
async def get_user_profile(username: str, current_user: dict = Depends(get_current_user)):
    """
    Get another user's public profile (for leaderboard, etc).
    """
    user = find_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == username
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Return only public information
    return {
        "username": user["username"],
        "created_at": user.get("created_at"),
        "stats": user.get("stats", {}),
        "role": user.get("role")
    }