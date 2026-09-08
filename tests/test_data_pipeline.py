import pytest
import os
import pandas as pd
from backend.src.data_ingestion import download_data
from backend.src.data_preprocessing import clean_and_prepare_survival_data

def test_pipeline_execution():
    raw_path = download_data()
    assert os.path.exists(raw_path), "Raw data file was not downloaded."
    
    df = clean_and_prepare_survival_data(input_path=raw_path)
    assert not df.empty, "Cleaned dataframe is empty."
    
    # Survival checks
    assert 'tenure_months' in df.columns, "Duration column missing."
    assert 'churn_event' in df.columns, "Event column missing."
    assert (df['tenure_months'] > 0).all(), "Tenure contains zero or negative values."
    assert df['churn_event'].isin([0, 1]).all(), "Event contains non-binary values."
    assert df['TotalCharges'].isnull().sum() == 0, "TotalCharges has unresolved nulls."