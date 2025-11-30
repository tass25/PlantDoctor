from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from typing import List, Optional
from utils.json_handler import find_all_json_items
from utils.jwt_handler import get_current_user
from config import settings

router = APIRouter(prefix="/leaderboard", tags=["Leaderboard"])

class LeaderboardUser(BaseModel):
    rank: int
    username: str
    total_points: int
    total_scans: int
    plants_saved: int
    badges_earned: int
    perfect_scans: int
    average_accuracy: float
    is_current_user: bool = False

class LeaderboardResponse(BaseModel):
    top_users: List[LeaderboardUser]
    current_user: Optional[LeaderboardUser]
    total_users: int

class GlobalStats(BaseModel):
    total_users: int
    total_analyses: int
    total_diseases_detected: int
    average_system_accuracy: float
    top_disease: str
    most_active_user: str

@router.get("/", response_model=LeaderboardResponse)
async def get_leaderboard(
    limit: int = Query(default=10, ge=1, le=100),
    sort_by: str = Query(default="points", regex="^(points|scans|plants_saved|badges)$"),
    current_user: dict = Depends(get_current_user)
):
    """
    Get global leaderboard with rankings.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    
    # Filter out password field
    users_data = [{k: v for k, v in user.items() if k != "password"} for user in all_users]
    
    # Sort based on criteria
    sort_mapping = {
        "points": lambda u: u.get("stats", {}).get("total_points", 0),
        "scans": lambda u: u.get("stats", {}).get("total_scans", 0),
        "plants_saved": lambda u: u.get("stats", {}).get("plants_saved", 0),
        "badges": lambda u: u.get("stats", {}).get("badges_earned", 0)
    }
    
    users_data.sort(key=sort_mapping[sort_by], reverse=True)
    
    # Create leaderboard entries
    leaderboard = []
    current_user_entry = None
    
    for idx, user in enumerate(users_data, 1):
        stats = user.get("stats", {})
        entry = LeaderboardUser(
            rank=idx,
            username=user["username"],
            total_points=stats.get("total_points", 0),
            total_scans=stats.get("total_scans", 0),
            plants_saved=stats.get("plants_saved", 0),
            badges_earned=stats.get("badges_earned", 0),
            perfect_scans=stats.get("perfect_scans", 0),
            average_accuracy=stats.get("average_accuracy", 0),
            is_current_user=(user["username"] == current_user["username"])
        )
        
        if idx <= limit:
            leaderboard.append(entry)
        
        if user["username"] == current_user["username"]:
            current_user_entry = entry
    
    return {
        "top_users": leaderboard,
        "current_user": current_user_entry,
        "total_users": len(users_data)
    }

@router.get("/stats", response_model=GlobalStats)
async def get_global_stats(current_user: dict = Depends(get_current_user)):
    """
    Get global platform statistics.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    all_history = find_all_json_items(settings.HISTORY_FILE)
    
    if not all_users or not all_history:
        return {
            "total_users": len(all_users),
            "total_analyses": 0,
            "total_diseases_detected": 0,
            "average_system_accuracy": 0,
            "top_disease": "N/A",
            "most_active_user": "N/A"
        }
    
    # Calculate statistics
    total_analyses = len(all_history)
    diseases_detected = sum(1 for h in all_history if h.get("best_disease", "").lower() != "healthy")
    
    total_confidence = sum(h.get("best_confidence", 0) for h in all_history)
    average_accuracy = round(total_confidence / total_analyses, 2) if total_analyses > 0 else 0
    
    # Find top disease
    disease_counts = {}
    for h in all_history:
        disease = h.get("best_disease", "Unknown")
        if disease.lower() != "healthy":
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    top_disease = max(disease_counts, key=disease_counts.get) if disease_counts else "N/A"
    
    # Find most active user
    user_scan_counts = {}
    for user in all_users:
        username = user.get("username")
        scan_count = user.get("stats", {}).get("total_scans", 0)
        user_scan_counts[username] = scan_count
    
    most_active_user = max(user_scan_counts, key=user_scan_counts.get) if user_scan_counts else "N/A"
    
    return {
        "total_users": len(all_users),
        "total_analyses": total_analyses,
        "total_diseases_detected": diseases_detected,
        "average_system_accuracy": average_accuracy,
        "top_disease": top_disease,
        "most_active_user": most_active_user
    }

@router.get("/top-performers")
async def get_top_performers(current_user: dict = Depends(get_current_user)):
    """
    Get various top performer lists for different categories.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    users_data = [{k: v for k, v in user.items() if k != "password"} for user in all_users]
    
    # Top by points
    top_points = sorted(
        users_data,
        key=lambda u: u.get("stats", {}).get("total_points", 0),
        reverse=True
    )[:5]
    
    # Top by plants saved
    top_plants_saved = sorted(
        users_data,
        key=lambda u: u.get("stats", {}).get("plants_saved", 0),
        reverse=True
    )[:5]
    
    # Top by accuracy
    top_accuracy = sorted(
        users_data,
        key=lambda u: u.get("stats", {}).get("average_accuracy", 0),
        reverse=True
    )[:5]
    
    # Top by badges
    top_badges = sorted(
        users_data,
        key=lambda u: u.get("stats", {}).get("badges_earned", 0),
        reverse=True
    )[:5]
    
    return {
        "top_points": [
            {
                "username": u["username"],
                "value": u.get("stats", {}).get("total_points", 0)
            }
            for u in top_points
        ],
        "top_plants_saved": [
            {
                "username": u["username"],
                "value": u.get("stats", {}).get("plants_saved", 0)
            }
            for u in top_plants_saved
        ],
        "top_accuracy": [
            {
                "username": u["username"],
                "value": u.get("stats", {}).get("average_accuracy", 0)
            }
            for u in top_accuracy
        ],
        "top_badges": [
            {
                "username": u["username"],
                "value": u.get("stats", {}).get("badges_earned", 0)
            }
            for u in top_badges
        ]
    }