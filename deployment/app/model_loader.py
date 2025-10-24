import os
import joblib
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd


class ModelLoader:
    """
    Model loader and manager for ML model serving
    """
    
    def __init__(self, model_path: str = "models/saved_model.pkl"):
        """
        Initialize model loader
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path = model_path
        self.model = None
        self.loaded_at = None
        
        # Load model on initialization
        self.load_model()
    
    def load_model(self):
        """Load the model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        self.loaded_at = datetime.now()
        print(f"Model loaded successfully at {self.loaded_at}")
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the loaded model
        
        Args:
            features: DataFrame with feature columns
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        predictions = self.model.predict(features)
        return predictions
    
    def get_model_version(self) -> str:
        """
        Get model version based on file modification time
        
        Returns:
            Model version string
        """
        if not os.path.exists(self.model_path):
            return "unknown"
        
        mod_time = os.path.getmtime(self.model_path)
        return datetime.fromtimestamp(mod_time).strftime("%Y%m%d_%H%M%S")
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_path": self.model_path,
            "model_type": str(type(self.model).__name__) if self.model else None,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "version": self.get_model_version()
        }
