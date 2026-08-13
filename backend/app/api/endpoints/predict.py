import os
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

try:
    from backend.app.services.model_service import model_service, CLASS_NAMES
except ImportError:
    from app.services.model_service import model_service, CLASS_NAMES

router = APIRouter()

@router.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": model_service.model_path is not None and os.path.exists(model_service.model_path),
        "model_path": model_service.model_path
    }

@router.get("/classes")
def get_supported_classes():
    return {
        "total_classes": len(CLASS_NAMES),
        "classes": CLASS_NAMES
    }

@router.get("/stats")
def get_system_stats():
    return {
        "disease_classes": 38,
        "training_images": "87,000+",
        "model_accuracy": "95%+",
        "inference_speed": "< 2 seconds",
        "supported_crops": [
            "Apple", "Blueberry", "Cherry", "Corn", "Grape", 
            "Orange", "Peach", "Pepper", "Potato", "Raspberry", 
            "Soybean", "Squash", "Strawberry", "Tomato"
        ]
    }

@router.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image (JPEG, PNG).")
    
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")
            
        prediction = model_service.predict(file_bytes)
        return JSONResponse(content=prediction)
    except ValueError as ve:
        raise HTTPException(status_code=500, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
