import os
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DATA_DIR = "data/processed"
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "telco_churn_clean.csv")

def clean_and_prepare_survival_data(input_path: str = "data/raw/telco_churn_raw.csv", output_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    logging.info(f"Loading raw data from {input_path}")
    df = pd.read_csv(input_path)
    
    # 1. Clean TotalCharges (strip whitespace, fill zero-tenure NaNs with 0.0)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].astype(str).str.strip(), errors='coerce').fillna(0.0)
    
    # 2. Binary encode Target & Survival Event: 1 = Churned, 0 = Censored (Active)
    df['churn_event'] = (df['Churn'] == 'Yes').astype(int)
    
    # 3. Rename & validate duration/tenure (T)
    df['tenure_months'] = df['tenure'].astype(int)
    
    # Check for invalid zero or negative tenures in survival calculations
    # In survival analysis, tenure=0 causes division by zero; add small epsilon or set min to 1
    df['tenure_months'] = df['tenure_months'].apply(lambda x: 1 if x == 0 else x)
    
    # 4. Standardize text columns (remove 'No internet service' redundancies)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No'})
        
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
    
    # 5. Save cleaned artifact
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Processed dataset saved to {output_path} | Shape: {df.shape}")
    return df

if __name__ == "__main__":
    clean_and_prepare_survival_data()