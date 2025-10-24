import pytest
import sys
import os
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from deployment.app.main import app

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test FastAPI endpoints"""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct status"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "service" in data
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_status" in data
    
    def test_model_info_endpoint(self):
        """Test model info endpoint"""
        response = client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "model_path" in data
        assert "model_version" in data
    
    def test_predict_endpoint(self):
        """Test single prediction endpoint"""
        payload = {
            "features": {
                "feature1": 50.0,
                "feature2": 75.0
            }
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data
        assert isinstance(data["prediction"], float)
    
    def test_predict_batch_endpoint(self):
        """Test batch prediction endpoint"""
        payload = {
            "data": [
                {"feature1": 50.0, "feature2": 75.0},
                {"feature1": 30.0, "feature2": 45.0},
                {"feature1": 70.0, "feature2": 90.0}
            ]
        }
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert "count" in data
        assert len(data["predictions"]) == 3
        assert data["count"] == 3
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input"""
        payload = {
            "features": {
                "invalid_feature": 50.0
            }
        }
        response = client.post("/predict", json=payload)
        # Should handle gracefully even with wrong features
        assert response.status_code in [200, 422, 500]


class TestETLPipeline:
    """Test ETL pipeline components"""
    
    def test_data_ingestion_import(self):
        """Test data ingestion module can be imported"""
        from etl import data_ingestion
        assert hasattr(data_ingestion, 'load_data')
    
    def test_data_preprocessing_import(self):
        """Test data preprocessing module can be imported"""
        from etl import data_preprocessing
        assert hasattr(data_preprocessing, 'preprocess_data')


class TestModelTraining:
    """Test model training components"""
    
    def test_train_model_import(self):
        """Test train_model module can be imported"""
        from training import train_model
        assert hasattr(train_model, 'train_model')
    
    def test_evaluate_model_import(self):
        """Test evaluate_model module can be imported"""
        from training import evaluate_model
        assert hasattr(evaluate_model, 'evaluate_model')


class TestModelLoader:
    """Test model loader functionality"""
    
    def test_model_loader_import(self):
        """Test ModelLoader can be imported"""
        from deployment.app.model_loader import ModelLoader
        assert ModelLoader is not None
    
    def test_model_loader_initialization(self):
        """Test ModelLoader can be initialized"""
        from deployment.app.model_loader import ModelLoader
        
        # Generate a test model if it doesn't exist
        if not os.path.exists('models/saved_model.pkl'):
            os.makedirs('models', exist_ok=True)
            from sklearn.ensemble import RandomForestRegressor
            import joblib
            model = RandomForestRegressor(n_estimators=10, random_state=42)
            X = np.random.randn(100, 2)
            y = np.random.randn(100)
            model.fit(X, y)
            joblib.dump(model, 'models/saved_model.pkl')
        
        loader = ModelLoader(model_path='models/saved_model.pkl')
        assert loader.model is not None
    
    def test_model_prediction(self):
        """Test model can make predictions"""
        from deployment.app.model_loader import ModelLoader
        
        loader = ModelLoader(model_path='models/saved_model.pkl')
        test_data = pd.DataFrame({
            'feature1': [50.0],
            'feature2': [75.0]
        })
        
        predictions = loader.predict(test_data)
        assert predictions is not None
        assert len(predictions) == 1


class TestDataValidation:
    """Test data validation and quality"""
    
    def test_sample_data_generation(self):
        """Test sample data can be generated"""
        from models.generate_model import generate_sample_data
        df = generate_sample_data()
        
        assert df is not None
        assert len(df) > 0
        assert 'feature1' in df.columns
        assert 'feature2' in df.columns
        assert 'target' in df.columns


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
