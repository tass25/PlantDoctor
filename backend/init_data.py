"""
Initialize JSON files with sample data for testing.
Run this script once before starting the server.
"""

import json
from datetime import datetime
from utils.hashing import hash_password

def init_users():
    """Initialize users.json with a test admin and user"""
    users = [
        {
            "id": "admin-001",
            "username": "admin",
            "password": hash_password("admin123"),
            "role": "admin",
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
        },
        {
            "id": "user-001",
            "username": "testuser",
            "password": hash_password("test123"),
            "role": "user",
            "created_at": datetime.utcnow().isoformat(),
            "stats": {
                "total_points": 150,
                "total_scans": 3,
                "plants_saved": 2,
                "diseases_found": 2,
                "healthy_plants": 1,
                "perfect_scans": 1,
                "average_accuracy": 88.5,
                "badges_earned": 2
            }
        }
    ]
    
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)
    print("✅ users.json initialized")

def init_history():
    """Initialize historique.json with empty array"""
    with open('historique.json', 'w') as f:
        json.dump([], f, indent=2)
    print("✅ historique.json initialized")

def init_badges():
    """Initialize badges.json with sample badges"""
    badges = [
        {
            "username": "testuser",
            "name": "First Steps",
            "emoji": "🌱",
            "unlocked_at": datetime.utcnow().isoformat(),
            "description": "Complete your first plant analysis"
        },
        {
            "username": "testuser",
            "name": "Eagle Eye",
            "emoji": "👁️",
            "unlocked_at": datetime.utcnow().isoformat(),
            "description": "Achieve 3 perfect scans (95%+ accuracy)"
        }
    ]
    
    with open('badges.json', 'w') as f:
        json.dump(badges, f, indent=2)
    print("✅ badges.json initialized")

def init_models():
    """Initialize models.json with model statistics"""
    models = {
        "Model 1": {
            "wins": 0,
            "total_confidence": 0,
            "count": 0,
            "average_confidence": 0
        },
        "Model 2": {
            "wins": 0,
            "total_confidence": 0,
            "count": 0,
            "average_confidence": 0
        }
    }
    
    with open('models.json', 'w') as f:
        json.dump(models, f, indent=2)
    print("✅ models.json initialized")

if __name__ == "__main__":
    print("🌿 Initializing PlantDoctor data files...\n")
    init_users()
    init_history()
    init_badges()
    init_models()
    print("\n✅ All data files initialized successfully!")
    print("\nTest credentials:")
    print("Admin - username: admin, password: admin123")
    print("User - username: testuser, password: test123")