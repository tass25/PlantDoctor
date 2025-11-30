from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from utils.json_handler import find_all_json_items, find_json_item
from utils.jwt_handler import get_current_user
from config import settings

router = APIRouter(prefix="/badges", tags=["Badges"])

class Badge(BaseModel):
    name: str
    emoji: str
    description: str
    unlocked_at: Optional[str] = None
    unlocked: bool = False
    progress: float = 0
    threshold: int = 0

class BadgeCollection(BaseModel):
    earned_badges: List[Badge]
    available_badges: List[Badge]
    total_badges: int
    completion_percentage: float

# Define all available badges with their criteria
AVAILABLE_BADGES = [
    {
        "name": "First Steps",
        "emoji": "🌱",
        "description": "Complete your first plant analysis",
        "threshold": 1,
        "stat_key": "total_scans"
    },
    {
        "name": "Plant Savior",
        "emoji": "🌿",
        "description": "Detect 5 plant diseases",
        "threshold": 5,
        "stat_key": "diseases_found"
    },
    {
        "name": "Eagle Eye",
        "emoji": "👁️",
        "description": "Achieve 3 perfect scans (95%+ accuracy)",
        "threshold": 3,
        "stat_key": "perfect_scans"
    },
    {
        "name": "Dedicated Gardener",
        "emoji": "🪴",
        "description": "Perform 10 plant analyses",
        "threshold": 10,
        "stat_key": "total_scans"
    },
    {
        "name": "Plant Doctor",
        "emoji": "⚕️",
        "description": "Save 25 plants from disease",
        "threshold": 25,
        "stat_key": "plants_saved"
    },
    {
        "name": "Guardian",
        "emoji": "🛡️",
        "description": "Save 10 plants",
        "threshold": 10,
        "stat_key": "plants_saved"
    },
    {
        "name": "Expert Botanist",
        "emoji": "🔬",
        "description": "Complete 50 plant analyses",
        "threshold": 50,
        "stat_key": "total_scans"
    },
    {
        "name": "Accuracy Master",
        "emoji": "🎯",
        "description": "Maintain 90%+ average accuracy over 20 scans",
        "threshold": 20,
        "stat_key": "total_scans",
        "extra_condition": lambda stats: stats.get("average_accuracy", 0) >= 90
    },
    {
        "name": "Century Club",
        "emoji": "💯",
        "description": "Perform 100 plant analyses",
        "threshold": 100,
        "stat_key": "total_scans"
    },
    {
        "name": "Disease Detective",
        "emoji": "🔍",
        "description": "Detect 50 different diseases",
        "threshold": 50,
        "stat_key": "diseases_found"
    },
    {
        "name": "Point Champion",
        "emoji": "⭐",
        "description": "Earn 1000 points",
        "threshold": 1000,
        "stat_key": "total_points"
    },
    {
        "name": "Elite Scorer",
        "emoji": "💎",
        "description": "Earn 5000 points",
        "threshold": 5000,
        "stat_key": "total_points"
    },
    {
        "name": "Master",
        "emoji": "👑",
        "description": "Earn 10000 points",
        "threshold": 10000,
        "stat_key": "total_points"
    },
    {
        "name": "Perfectionist",
        "emoji": "✨",
        "description": "Achieve 10 perfect scans",
        "threshold": 10,
        "stat_key": "perfect_scans"
    },
    {
        "name": "Environmental Hero",
        "emoji": "🌍",
        "description": "Save 100 plants",
        "threshold": 100,
        "stat_key": "plants_saved"
    }
]

@router.get("/", response_model=BadgeCollection)
async def get_user_badges(current_user: dict = Depends(get_current_user)):
    """
    Get all badges for current user - earned and available.
    """
    # Get user stats
    user = find_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == current_user["username"]
    )
    
    if not user:
        return {
            "earned_badges": [],
            "available_badges": [],
            "total_badges": len(AVAILABLE_BADGES),
            "completion_percentage": 0
        }
    
    stats = user.get("stats", {})
    
    # Get earned badges from database
    earned_badges_data = find_all_json_items(
        settings.BADGES_FILE,
        lambda b: b.get("username") == current_user["username"]
    )
    
    earned_badge_names = set(b.get("name") for b in earned_badges_data)
    
    earned_badges = []
    available_badges = []
    
    for badge_def in AVAILABLE_BADGES:
        stat_value = stats.get(badge_def["stat_key"], 0)
        progress = min((stat_value / badge_def["threshold"]) * 100, 100)
        
        # Check extra condition if exists
        is_unlocked = stat_value >= badge_def["threshold"]
        if "extra_condition" in badge_def and is_unlocked:
            is_unlocked = badge_def["extra_condition"](stats)
        
        # Find unlock date if badge is earned
        unlocked_at = None
        if badge_def["name"] in earned_badge_names:
            earned_badge = next(
                (b for b in earned_badges_data if b.get("name") == badge_def["name"]),
                None
            )
            if earned_badge:
                unlocked_at = earned_badge.get("unlocked_at")
        
        badge = Badge(
            name=badge_def["name"],
            emoji=badge_def["emoji"],
            description=badge_def["description"],
            unlocked_at=unlocked_at,
            unlocked=is_unlocked,
            progress=round(progress, 1),
            threshold=badge_def["threshold"]
        )
        
        if is_unlocked:
            earned_badges.append(badge)
        else:
            available_badges.append(badge)
    
    total_badges = len(AVAILABLE_BADGES)
    completion_percentage = round((len(earned_badges) / total_badges) * 100, 1) if total_badges > 0 else 0
    
    return {
        "earned_badges": earned_badges,
        "available_badges": available_badges,
        "total_badges": total_badges,
        "completion_percentage": completion_percentage
    }

@router.get("/progress")
async def get_badge_progress(current_user: dict = Depends(get_current_user)):
    """
    Get detailed progress for all badges.
    """
    user = find_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == current_user["username"]
    )
    
    if not user:
        return {"badges": []}
    
    stats = user.get("stats", {})
    
    badge_progress = []
    for badge_def in AVAILABLE_BADGES:
        stat_value = stats.get(badge_def["stat_key"], 0)
        progress = min((stat_value / badge_def["threshold"]) * 100, 100)
        remaining = max(badge_def["threshold"] - stat_value, 0)
        
        badge_progress.append({
            "name": badge_def["name"],
            "emoji": badge_def["emoji"],
            "description": badge_def["description"],
            "current_value": stat_value,
            "threshold": badge_def["threshold"],
            "progress_percentage": round(progress, 1),
            "remaining": remaining,
            "unlocked": stat_value >= badge_def["threshold"]
        })
    
    return {"badges": badge_progress}

@router.get("/leaderboard")
async def get_badge_leaderboard(current_user: dict = Depends(get_current_user)):
    """
    Get users ranked by badge count.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    
    # Sort by badges earned
    users_with_badges = sorted(
        all_users,
        key=lambda u: u.get("stats", {}).get("badges_earned", 0),
        reverse=True
    )[:10]
    
    leaderboard = []
    for idx, user in enumerate(users_with_badges, 1):
        leaderboard.append({
            "rank": idx,
            "username": user["username"],
            "badges_earned": user.get("stats", {}).get("badges_earned", 0),
            "is_current_user": user["username"] == current_user["username"]
        })
    
    return {"leaderboard": leaderboard}