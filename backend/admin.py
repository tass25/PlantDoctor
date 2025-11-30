from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
from utils.json_handler import find_all_json_items, find_json_item, update_json_item, delete_json_item
from utils.jwt_handler import require_admin
from config import settings

router = APIRouter(prefix="/admin", tags=["Admin"], dependencies=[Depends(require_admin)])

class AdminDashboardStats(BaseModel):
    total_users: int
    total_scans: int
    diseases_found: int
    system_accuracy: float
    active_today: int
    user_growth: dict
    activity_heatmap: dict
    top_users: List[dict]
    disease_distribution: dict
    model_performance: dict

class UserManagementResponse(BaseModel):
    total_users: int
    active_users: int
    total_badges: int
    average_scans: float
    users: List[dict]

class SystemStatistics(BaseModel):
    badge_distribution: dict
    points_distribution: dict
    disease_frequency: dict
    model_statistics: dict
    activity_trends: dict
    environmental_impact: dict
    health_metrics: dict

@router.get("/dashboard", response_model=AdminDashboardStats)
async def get_admin_dashboard():
    """
    Get comprehensive admin dashboard statistics.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    all_history = find_all_json_items(settings.HISTORY_FILE)
    
    # Basic metrics
    total_users = len(all_users)
    total_scans = len(all_history)
    diseases_found = sum(1 for h in all_history if h.get("best_disease", "").lower() != "healthy")
    
    total_confidence = sum(h.get("best_confidence", 0) for h in all_history)
    system_accuracy = round(total_confidence / total_scans, 2) if total_scans > 0 else 0
    
    # Active users today
    today = datetime.utcnow().date()
    active_today = len(set(
        h.get("username") for h in all_history
        if datetime.fromisoformat(h.get("timestamp", "")).date() == today
    ))
    
    # User growth (last 30 days)
    user_growth = calculate_user_growth(all_users)
    
    # Activity heatmap (last 7 days)
    activity_heatmap = calculate_activity_heatmap(all_history)
    
    # Top users by points
    top_users = sorted(
        all_users,
        key=lambda u: u.get("stats", {}).get("total_points", 0),
        reverse=True
    )[:10]
    
    top_users_data = [
        {
            "username": u["username"],
            "points": u.get("stats", {}).get("total_points", 0),
            "scans": u.get("stats", {}).get("total_scans", 0),
            "accuracy": u.get("stats", {}).get("average_accuracy", 0)
        }
        for u in top_users
    ]
    
    # Disease distribution
    disease_counts = {}
    for h in all_history:
        disease = h.get("best_disease", "Unknown")
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Model performance
    model1_wins = sum(1 for h in all_history if h.get("best_model") == "Model 1")
    model2_wins = sum(1 for h in all_history if h.get("best_model") == "Model 2")
    
    model_performance = {
        "Model 1": {
            "wins": model1_wins,
            "percentage": round((model1_wins / total_scans) * 100, 2) if total_scans > 0 else 0
        },
        "Model 2": {
            "wins": model2_wins,
            "percentage": round((model2_wins / total_scans) * 100, 2) if total_scans > 0 else 0
        }
    }
    
    return {
        "total_users": total_users,
        "total_scans": total_scans,
        "diseases_found": diseases_found,
        "system_accuracy": system_accuracy,
        "active_today": active_today,
        "user_growth": user_growth,
        "activity_heatmap": activity_heatmap,
        "top_users": top_users_data,
        "disease_distribution": disease_counts,
        "model_performance": model_performance
    }

@router.get("/users", response_model=UserManagementResponse)
async def get_all_users(
    search: Optional[str] = Query(None),
    sort_by: str = Query(default="points", regex="^(points|scans|badges|created_at)$"),
    order: str = Query(default="desc", regex="^(asc|desc)$")
):
    """
    Get all users with filtering and sorting options.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    all_badges = find_all_json_items(settings.BADGES_FILE)
    
    # Calculate metrics
    total_users = len(all_users)
    
    # Active users (at least 1 scan)
    active_users = sum(1 for u in all_users if u.get("stats", {}).get("total_scans", 0) > 0)
    
    # Total badges across all users
    total_badges_count = len(all_badges)
    
    # Average scans
    total_scans = sum(u.get("stats", {}).get("total_scans", 0) for u in all_users)
    average_scans = round(total_scans / total_users, 2) if total_users > 0 else 0
    
    # Filter users if search query provided
    if search:
        all_users = [u for u in all_users if search.lower() in u.get("username", "").lower()]
    
    # Sort users
    sort_mapping = {
        "points": lambda u: u.get("stats", {}).get("total_points", 0),
        "scans": lambda u: u.get("stats", {}).get("total_scans", 0),
        "badges": lambda u: u.get("stats", {}).get("badges_earned", 0),
        "created_at": lambda u: u.get("created_at", "")
    }
    
    reverse = (order == "desc")
    all_users.sort(key=sort_mapping[sort_by], reverse=reverse)
    
    # Format user data (remove passwords)
    users_data = []
    for user in all_users:
        user_data = {k: v for k, v in user.items() if k != "password"}
        users_data.append(user_data)
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "total_badges": total_badges_count,
        "average_scans": average_scans,
        "users": users_data
    }

@router.get("/users/{username}")
async def get_user_details(username: str):
    """
    Get detailed information about a specific user.
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
    
    # Get user's history
    user_history = find_all_json_items(
        settings.HISTORY_FILE,
        lambda h: h.get("username") == username
    )
    
    # Get user's badges
    user_badges = find_all_json_items(
        settings.BADGES_FILE,
        lambda b: b.get("username") == username
    )
    
    # Calculate additional analytics
    accuracy_over_time = []
    for h in sorted(user_history, key=lambda x: x.get("timestamp", "")):
        accuracy_over_time.append({
            "timestamp": h.get("timestamp"),
            "accuracy": h.get("best_confidence")
        })
    
    disease_distribution = {}
    for h in user_history:
        disease = h.get("best_disease", "Unknown")
        disease_distribution[disease] = disease_distribution.get(disease, 0) + 1
    
    # Remove password from response
    user_data = {k: v for k, v in user.items() if k != "password"}
    
    return {
        "user": user_data,
        "history_count": len(user_history),
        "badges": user_badges,
        "analytics": {
            "accuracy_over_time": accuracy_over_time,
            "disease_distribution": disease_distribution
        }
    }

@router.delete("/users/{username}")
async def delete_user(username: str):
    """
    Delete a user and all their associated data.
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
    
    # Prevent deleting admin users
    if user.get("role") == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete admin users"
        )
    
    # Delete user
    delete_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == username
    )
    
    # Delete user's history
    delete_json_item(
        settings.HISTORY_FILE,
        lambda h: h.get("username") == username
    )
    
    # Delete user's badges
    delete_json_item(
        settings.BADGES_FILE,
        lambda b: b.get("username") == username
    )
    
    return {"message": f"User {username} deleted successfully"}

@router.get("/statistics", response_model=SystemStatistics)
async def get_system_statistics():
    """
    Get comprehensive system statistics.
    """
    all_users = find_all_json_items(settings.USERS_FILE)
    all_history = find_all_json_items(settings.HISTORY_FILE)
    all_badges = find_all_json_items(settings.BADGES_FILE)
    
    # Badge distribution
    badge_counts = {}
    for badge in all_badges:
        name = badge.get("name", "Unknown")
        badge_counts[name] = badge_counts.get(name, 0) + 1
    
    # Points distribution (buckets)
    points_distribution = {
        "0-100": 0,
        "101-500": 0,
        "501-1000": 0,
        "1001-5000": 0,
        "5000+": 0
    }
    
    for user in all_users:
        points = user.get("stats", {}).get("total_points", 0)
        if points <= 100:
            points_distribution["0-100"] += 1
        elif points <= 500:
            points_distribution["101-500"] += 1
        elif points <= 1000:
            points_distribution["501-1000"] += 1
        elif points <= 5000:
            points_distribution["1001-5000"] += 1
        else:
            points_distribution["5000+"] += 1
    
    # Disease frequency and severity
    disease_stats = {}
    for h in all_history:
        disease = h.get("best_disease", "Unknown")
        urgency = h.get("urgency_level", "low")
        
        if disease not in disease_stats:
            disease_stats[disease] = {
                "count": 0,
                "high_urgency": 0,
                "medium_urgency": 0,
                "low_urgency": 0
            }
        
        disease_stats[disease]["count"] += 1
        disease_stats[disease][f"{urgency}_urgency"] += 1
    
    # Model statistics
    model_stats = calculate_model_statistics(all_history)
    
    # Activity trends (last 7 days)
    activity_trends = calculate_weekly_activity(all_history)
    
    # Environmental impact
    total_plants_saved = sum(u.get("stats", {}).get("plants_saved", 0) for u in all_users)
    environmental_impact = {
        "plants_saved": total_plants_saved,
        "water_saved": total_plants_saved * 50,
        "co2_offset": round(total_plants_saved * 2.5, 2),
        "pesticide_reduced": round(total_plants_saved * 0.3, 2)
    }
    
    # Health metrics
    retention_rate = calculate_retention_rate(all_users, all_history)
    engagement_score = calculate_engagement_score(all_users)
    
    health_metrics = {
        "retention_rate": retention_rate,
        "system_accuracy": round(
            sum(h.get("best_confidence", 0) for h in all_history) / len(all_history), 2
        ) if all_history else 0,
        "engagement_score": engagement_score
    }
    
    return {
        "badge_distribution": badge_counts,
        "points_distribution": points_distribution,
        "disease_frequency": disease_stats,
        "model_statistics": model_stats,
        "activity_trends": activity_trends,
        "environmental_impact": environmental_impact,
        "health_metrics": health_metrics
    }

@router.get("/recent-activity")
async def get_recent_activity(limit: int = Query(default=10, ge=1, le=100)):
    """
    Get recent system activity (last analyses).
    """
    all_history = find_all_json_items(settings.HISTORY_FILE)
    
    # Sort by timestamp descending
    all_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    recent = all_history[:limit]
    
    activity_feed = [
        {
            "username": h.get("username"),
            "disease": h.get("best_disease"),
            "confidence": h.get("best_confidence"),
            "timestamp": h.get("timestamp"),
            "urgency": h.get("urgency_level"),
            "model": h.get("best_model")
        }
        for h in recent
    ]
    
    return {"activity": activity_feed}

# Helper functions
def calculate_user_growth(users: list) -> dict:
    """Calculate user growth over last 30 days"""
    growth = {}
    now = datetime.utcnow()
    
    for i in range(30):
        date = (now - timedelta(days=i)).date()
        date_str = date.isoformat()
        
        count = sum(
            1 for u in users
            if datetime.fromisoformat(u.get("created_at", "")).date() <= date
        )
        growth[date_str] = count
    
    return growth

def calculate_activity_heatmap(history: list) -> dict:
    """Calculate activity heatmap for last 7 days"""
    heatmap = {}
    now = datetime.utcnow()
    
    for i in range(7):
        date = (now - timedelta(days=i)).date()
        date_str = date.isoformat()
        
        count = sum(
            1 for h in history
            if datetime.fromisoformat(h.get("timestamp", "")).date() == date
        )
        heatmap[date_str] = count
    
    return heatmap

def calculate_model_statistics(history: list) -> dict:
    """Calculate detailed model statistics"""
    model_stats = {
        "Model 1": {"wins": 0, "total_confidence": 0, "count": 0},
        "Model 2": {"wins": 0, "total_confidence": 0, "count": 0}
    }
    
    for h in history:
        model = h.get("best_model")
        confidence = h.get("best_confidence", 0)
        
        if model in model_stats:
            model_stats[model]["wins"] += 1
            model_stats[model]["total_confidence"] += confidence
            model_stats[model]["count"] += 1
    
    # Calculate average confidence
    for model in model_stats:
        if model_stats[model]["count"] > 0:
            model_stats[model]["average_confidence"] = round(
                model_stats[model]["total_confidence"] / model_stats[model]["count"],
                2
            )
        else:
            model_stats[model]["average_confidence"] = 0
    
    return model_stats

def calculate_weekly_activity(history: list) -> dict:
    """Calculate activity for each day of the week"""
    weekly = {
        "Monday": 0, "Tuesday": 0, "Wednesday": 0,
        "Thursday": 0, "Friday": 0, "Saturday": 0, "Sunday": 0
    }
    
    for h in history:
        try:
            timestamp = datetime.fromisoformat(h.get("timestamp", ""))
            day_name = timestamp.strftime("%A")
            weekly[day_name] += 1
        except:
            pass
    
    return weekly

def calculate_retention_rate(users: list, history: list) -> float:
    """Calculate user retention rate (users active in last 7 days)"""
    if not users:
        return 0
    
    now = datetime.utcnow()
    week_ago = now - timedelta(days=7)
    
    active_users = set(
        h.get("username") for h in history
        if datetime.fromisoformat(h.get("timestamp", "")) >= week_ago
    )
    
    return round((len(active_users) / len(users)) * 100, 2)

def calculate_engagement_score(users: list) -> float:
    """Calculate overall engagement score"""
    if not users:
        return 0
    
    total_score = 0
    for user in users:
        stats = user.get("stats", {})
        score = (
            stats.get("total_scans", 0) * 1 +
            stats.get("badges_earned", 0) * 10 +
            stats.get("perfect_scans", 0) * 5
        )
        total_score += score
    
    return round(total_score / len(users), 2)