import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from etl.data_ingestion import load_data
from etl.data_preprocessing import preprocess_data


class TestDataIngestion:
    """Test data ingestion functionality"""
    
    def test_load_data_creates_file(self, tmp_path):
        """Test that load_data creates output file"""
        # Create sample CSV
        input_file = tmp_path / "test_input.csv"
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [7, 8, 9]
        })
        df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "test_output.parquet"
        
        # Run load_data
        result_df = load_data(str(input_file), str(output_file))
        
        assert os.path.exists(output_file)
        assert len(result_df) == 3
    
    def test_load_data_removes_duplicates(self, tmp_path):
        """Test that load_data removes duplicate rows"""
        input_file = tmp_path / "test_duplicates.csv"
        df = pd.DataFrame({
            'feature1': [1, 1, 2],
            'feature2': [4, 4, 5],
            'target': [7, 7, 8]
        })
        df.to_csv(input_file, index=False)
        
        output_file = tmp_path / "test_output.parquet"
        result_df = load_data(str(input_file), str(output_file))
        
        assert len(result_df) == 2  # One duplicate removed


class TestDataPreprocessing:
    """Test data preprocessing functionality"""
    
    def test_preprocess_scales_features(self, tmp_path):
        """Test that preprocessing scales features"""
        # Create sample data
        input_file = tmp_path / "test_input.parquet"
        df = pd.DataFrame({
            'feature1': [10, 20, 30],
            'feature2': [100, 200, 300],
            'target': [1, 2, 3]
        })
        df.to_parquet(input_file, index=False)
        
        output_file = tmp_path / "test_output.parquet"
        
        # Run preprocessing
        result_df = preprocess_data(str(input_file), str(output_file))
        
        # Check that features are scaled (mean ~0, std ~1)
        assert abs(result_df['feature1'].mean()) < 0.1
        assert abs(result_df['feature2'].mean()) < 0.1
