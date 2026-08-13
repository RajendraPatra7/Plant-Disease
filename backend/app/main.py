import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.api.endpoints.predict import router as predict_router

app = FastAPI(
    title="Smart Spray X API",
    description="AI-Driven Plant Disease Recognition & Pesticide Optimization System API",
    version="2.0.0"
)

# Enable CORS for React frontend (development & production origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vite dev server & production builds
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(predict_router, prefix="/api/v1")

@app.get("/")
def root_route():
    return {
        "message": "Welcome to Smart Spray X API Engine 🌿",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
