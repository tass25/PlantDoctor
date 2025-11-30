from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import settings
import auth
import users
import history
import leaderboard
import badges
import admin
from pydantic import BaseModel
from models import predict_image

# Initialize FastAPI app
app = FastAPI(
    title="PlantDoctor API",
    description="AI-powered plant disease detection system with gamification",
    version="1.0.0"
)

# Configure CORS - MUST be before route includes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Include routers
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(history.router)
app.include_router(leaderboard.router)
app.include_router(badges.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    """
    Root endpoint - API health check
    """
    return {
        "message": "PlantDoctor API is running! 🌿",
        "version": "1.0.0",
        "endpoints": {
            "auth": "/auth",
            "users": "/users",
            "history": "/history",
            "leaderboard": "/leaderboard",
            "badges": "/badges",
            "admin": "/admin"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "service": "PlantDoctor API"
    }


class PredictRequest(BaseModel):
    image_base64: str

@app.post("/predict")
async def predict(req: PredictRequest):
    return predict_image(req.image_base64)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)