from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_loader import ModelLoader

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise MLOps Prediction API",
    description="Production-grade ML model serving with FastAPI",
    version="1.0.0"
)

# Initialize model loader
model_loader = ModelLoader(model_path="models/saved_model.pkl")


class PredictionInput(BaseModel):
    """Schema for single prediction request"""
    features: Dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "feature1": 50.0,
                    "feature2": 75.0
                }
            }
        }


class BatchPredictionInput(BaseModel):
    """Schema for batch prediction request"""
    data: List[Dict[str, float]]

    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"feature1": 50.0, "feature2": 75.0},
                    {"feature1": 30.0, "feature2": 45.0}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    prediction: float
    model_version: str


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    predictions: List[float]
    model_version: str
    count: int


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Enterprise MLOps Prediction API",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Detailed health check"""
    model_status = "loaded" if model_loader.model is not None else "not_loaded"
    return {
        "status": "healthy",
        "model_status": model_status,
        "model_path": model_loader.model_path
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: PredictionInput):
    """
    Single prediction endpoint
    
    Args:
        input_data: Dictionary of feature names and values
        
    Returns:
        Prediction result and model version
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data.features])
        
        # Make prediction
        prediction = model_loader.predict(df)
        
        return PredictionResponse(
            prediction=float(prediction[0]),
            model_version=model_loader.get_model_version()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(input_data: BatchPredictionInput):
    """
    Batch prediction endpoint
    
    Args:
        input_data: List of feature dictionaries
        
    Returns:
        List of predictions and model version
    """
    try:
        # Convert input to DataFrame
        df = pd.DataFrame(input_data.data)
        
        # Make predictions
        predictions = model_loader.predict(df)
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in predictions],
            model_version=model_loader.get_model_version(),
            count=len(predictions)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.get("/model/info")
def model_info():
    """Get model metadata and information"""
    return {
        "model_path": model_loader.model_path,
        "model_version": model_loader.get_model_version(),
        "model_type": str(type(model_loader.model).__name__) if model_loader.model else None,
        "status": "loaded" if model_loader.model else "not_loaded"
    }


@app.post("/model/reload")
def reload_model():
    """Reload the model from disk"""
    try:
        model_loader.load_model()
        return {
            "status": "success",
            "message": "Model reloaded successfully",
            "model_version": model_loader.get_model_version()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model reload error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
