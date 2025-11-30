from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from utils.json_handler import find_all_json_items, append_json, find_json_item, update_json_item
from utils.jwt_handler import get_current_user
from config import settings
import uuid

router = APIRouter(prefix="/history", tags=["History"])

class Prediction(BaseModel):
    disease_name: str
    confidence: float
    model: str

class AnalysisRequest(BaseModel):
    image_url: str
    user_context: Optional[str] = None
    predictions: List[Prediction]
    best_model: str
    best_disease: str
    best_confidence: float
    urgency_level: str = Field(..., pattern="^(low|medium|high)$")

class AnalysisResponse(BaseModel):
    id: str
    username: str
    timestamp: str
    image_url: str
    user_context: Optional[str]
    predictions: List[dict]
    best_model: str
    best_disease: str
    best_confidence: float
    urgency_level: str
    points_earned: int
    badges_unlocked: List[dict]

class HistoryStats(BaseModel):
    total_analyses: int
    plant_types: int
    diseases_found: int
    healthy_plants: int
    average_accuracy: float
    model_performance: dict
    eco_metrics: dict

@router.post("/analysis", response_model=AnalysisResponse)
async def add_analysis(data: AnalysisRequest, current_user: dict = Depends(get_current_user)):
    """
    Add a new plant analysis to history and update user stats.
    """
    # Calculate points earned
    base_points = int(data.best_confidence)
    bonus_points = 0
    
    # High accuracy bonus
    if data.best_confidence >= 95:
        bonus_points += 50
    elif data.best_confidence >= 90:
        bonus_points += 30
    elif data.best_confidence >= 85:
        bonus_points += 15
    
    # Urgency bonus
    if data.urgency_level == "high":
        bonus_points += 25
    elif data.urgency_level == "medium":
        bonus_points += 15
    
    total_points = base_points + bonus_points
    
    # Create analysis record
    analysis_id = str(uuid.uuid4())
    analysis = {
        "id": analysis_id,
        "username": current_user["username"],
        "timestamp": datetime.utcnow().isoformat(),
        "image_url": data.image_url,
        "user_context": data.user_context,
        "predictions": [p.dict() for p in data.predictions],
        "best_model": data.best_model,
        "best_disease": data.best_disease,
        "best_confidence": data.best_confidence,
        "urgency_level": data.urgency_level,
        "points_earned": total_points
    }
    
    # Save to history
    append_json(settings.HISTORY_FILE, analysis)
    
    # Update user stats
    user = find_json_item(
        settings.USERS_FILE,
        lambda u: u.get("username") == current_user["username"]
    )
    
    if user:
        stats = user.get("stats", {})
        stats["total_scans"] = stats.get("total_scans", 0) + 1
        stats["total_points"] = stats.get("total_points", 0) + total_points
        
        if data.best_disease.lower() != "healthy":
            stats["diseases_found"] = stats.get("diseases_found", 0) + 1
        else:
            stats["healthy_plants"] = stats.get("healthy_plants", 0) + 1
        
        if data.best_confidence >= 95:
            stats["perfect_scans"] = stats.get("perfect_scans", 0) + 1
        
        # Calculate new average accuracy
        user_history = find_all_json_items(
            settings.HISTORY_FILE,
            lambda h: h.get("username") == current_user["username"]
        )
        total_accuracy = sum(h.get("best_confidence", 0) for h in user_history)
        stats["average_accuracy"] = round(total_accuracy / len(user_history), 2) if user_history else 0
        
        # Update user
        update_json_item(
            settings.USERS_FILE,
            lambda u: u.get("username") == current_user["username"],
            {"stats": stats}
        )
    
    # Check for badge unlocks
    badges_unlocked = check_badge_unlocks(current_user["username"], stats)
    
    # Update model stats
    update_model_stats(data.best_model, data.best_confidence)
    
    return {
        **analysis,
        "badges_unlocked": badges_unlocked
    }

@router.get("/", response_model=List[AnalysisResponse])
async def get_user_history(
    limit: Optional[int] = None,
    current_user: dict = Depends(get_current_user)
):
    """
    Get user's analysis history.
    """
    user_history = find_all_json_items(
        settings.HISTORY_FILE,
        lambda h: h.get("username") == current_user["username"]
    )
    
    # Sort by timestamp descending
    user_history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    if limit:
        user_history = user_history[:limit]
    
    # Add badges_unlocked field (empty for historical analyses)
    for analysis in user_history:
        if "badges_unlocked" not in analysis:
            analysis["badges_unlocked"] = []
    
    return user_history

@router.get("/stats", response_model=HistoryStats)
async def get_history_stats(current_user: dict = Depends(get_current_user)):
    """
    Get aggregated statistics from user's history.
    """
    user_history = find_all_json_items(
        settings.HISTORY_FILE,
        lambda h: h.get("username") == current_user["username"]
    )
    
    if not user_history:
        return {
            "total_analyses": 0,
            "plant_types": 0,
            "diseases_found": 0,
            "healthy_plants": 0,
            "average_accuracy": 0,
            "model_performance": {"Model 1": 0, "Model 2": 0},
            "eco_metrics": {
                "plants_saved": 0,
                "water_saved": 0,
                "co2_offset": 0,
                "pesticide_reduced": 0
            }
        }
    
    total_analyses = len(user_history)
    diseases = set(h.get("best_disease") for h in user_history)
    plant_types = len(diseases)
    diseases_found = sum(1 for h in user_history if h.get("best_disease", "").lower() != "healthy")
    healthy_plants = total_analyses - diseases_found
    
    total_confidence = sum(h.get("best_confidence", 0) for h in user_history)
    average_accuracy = round(total_confidence / total_analyses, 2) if total_analyses > 0 else 0
    
    # Model performance
    model1_wins = sum(1 for h in user_history if h.get("best_model") == "Model 1")
    model2_wins = sum(1 for h in user_history if h.get("best_model") == "Model 2")
    
    # Eco metrics (simplified calculations)
    plants_saved = diseases_found
    water_saved = plants_saved * 50  # liters
    co2_offset = plants_saved * 2.5  # kg
    pesticide_reduced = plants_saved * 0.3  # kg
    
    return {
        "total_analyses": total_analyses,
        "plant_types": plant_types,
        "diseases_found": diseases_found,
        "healthy_plants": healthy_plants,
        "average_accuracy": average_accuracy,
        "model_performance": {
            "Model 1": model1_wins,
            "Model 2": model2_wins
        },
        "eco_metrics": {
            "plants_saved": plants_saved,
            "water_saved": water_saved,
            "co2_offset": round(co2_offset, 2),
            "pesticide_reduced": round(pesticide_reduced, 2)
        }
    }

def check_badge_unlocks(username: str, stats: dict) -> List[dict]:
    """
    Check if user unlocked any new badges based on their stats.
    """
    badges_unlocked = []
    
    badge_criteria = [
        {"name": "First Steps", "emoji": "🌱", "condition": stats.get("total_scans", 0) >= 1, "threshold": 1},
        {"name": "Plant Savior", "emoji": "🌿", "condition": stats.get("diseases_found", 0) >= 5, "threshold": 5},
        {"name": "Eagle Eye", "emoji": "👁️", "condition": stats.get("perfect_scans", 0) >= 3, "threshold": 3},
        {"name": "Expert Botanist", "emoji": "🔬", "condition": stats.get("total_scans", 0) >= 50, "threshold": 50},
        {"name": "Guardian", "emoji": "🛡️", "condition": stats.get("plants_saved", 0) >= 10, "threshold": 10},
        {"name": "Master", "emoji": "👑", "condition": stats.get("total_points", 0) >= 5000, "threshold": 5000},
    ]
    
    existing_badges = find_all_json_items(
        settings.BADGES_FILE,
        lambda b: b.get("username") == username
    )
    existing_badge_names = [b.get("name") for b in existing_badges]
    
    for badge in badge_criteria:
        if badge["condition"] and badge["name"] not in existing_badge_names:
            new_badge = {
                "username": username,
                "name": badge["name"],
                "emoji": badge["emoji"],
                "unlocked_at": datetime.utcnow().isoformat(),
                "description": f"Achieved {badge['name']} milestone"
            }
            append_json(settings.BADGES_FILE, new_badge)
            badges_unlocked.append(new_badge)
            
            # Update user badge count
            update_json_item(
                settings.USERS_FILE,
                lambda u: u.get("username") == username,
                {"stats.badges_earned": len(existing_badges) + len(badges_unlocked)}
            )
    
    return badges_unlocked

def update_model_stats(model_name: str, confidence: float):
    """
    Update model performance statistics.
    """
    from utils.json_handler import read_json, write_json
    
    models_data = read_json(settings.MODELS_FILE)
    
    if not models_data or not isinstance(models_data, dict):
        models_data = {
            "Model 1": {"wins": 0, "total_confidence": 0, "count": 0},
            "Model 2": {"wins": 0, "total_confidence": 0, "count": 0}
        }
    
    if model_name in models_data:
        models_data[model_name]["wins"] = models_data[model_name].get("wins", 0) + 1
        models_data[model_name]["total_confidence"] = models_data[model_name].get("total_confidence", 0) + confidence
        models_data[model_name]["count"] = models_data[model_name].get("count", 0) + 1
    
    write_json(settings.MODELS_FILE, models_data)