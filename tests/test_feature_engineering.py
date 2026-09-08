import pytest
import os
import pandas as pd
from backend.src.feature_engineering import engineer_features, FEATURES_DATA_PATH, SCALER_PATH

def test_feature_engineering_pipeline():
    df_features = engineer_features()
    
    assert os.path.exists(FEATURES_DATA_PATH), "Feature dataset file was not created."
    assert os.path.exists(SCALER_PATH), "Scaler artifact was not saved."
    assert not df_features.empty, "Engineered feature dataframe is empty."
    
    # Check survival target columns remain intact
    assert 'tenure_months' in df_features.columns
    assert 'churn_event' in df_features.columns
    
    # Verify all columns are strictly numerical for survival algorithms
    non_numeric_cols = df_features.select_dtypes(exclude=['number']).columns.tolist()
    assert len(non_numeric_cols) == 0, f"Found non-numeric columns: {non_numeric_cols}"
    
    # Check no NaN values exist in engineered set
    assert df_features.isnull().sum().sum() == 0, "Engineered features contain NaN values."