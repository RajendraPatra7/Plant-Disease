import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Support both local (python -m uvicorn app.main:app) and
# Docker/Render (python -m uvicorn backend.app.main:app) import paths
try:
    from backend.app.api.endpoints.predict import router as predict_router
except ImportError:
    from app.api.endpoints.predict import router as predict_router

app = FastAPI(
    title="Smart Spray X API",
    description="AI-Driven Plant Disease Recognition & Pesticide Optimization System",
    version="2.0.0"
)

# CORS — allow Vercel frontend + local dev
allowed_origins = [
    "http://localhost:5173",                        # Vite dev server
    "http://localhost:3000",                        # Alt dev port
    "https://smartspray-x.vercel.app",              # Production frontend
]

# Also read from FRONTEND_URL env variable (set in Render dashboard)
frontend_url = os.environ.get("FRONTEND_URL")
if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
def root_route():
    return {
        "message": "Welcome to Smart Spray X API Engine 🌿",
        "docs": "/docs",
        "health": "/api/v1/health",
        "version": "2.0.0"
    }
