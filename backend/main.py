from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import auth, users, history, leaderboard, badges, admin

app = FastAPI(title="PlantDoctor Backend")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(history.router, prefix="/api/history", tags=["history"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["leaderboard"])
app.include_router(badges.router, prefix="/api/badges", tags=["badges"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])



