from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    JWT_SECRET: str = os.getenv("JWT_SECRET", "defaultsecret")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    
    # File paths
    USERS_FILE = "users.json"
    HISTORY_FILE = "historique.json"
    BADGES_FILE = "badges.json"
    MODELS_FILE = "models.json"

settings = Settings()