import os
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROCESSED_DATA_PATH = "data/processed/telco_churn_clean.csv"
FEATURES_DATA_PATH = "data/processed/telco_churn_features.csv"
SCALER_PATH = "data/processed/scaler.pkl"

CATEGORICAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_COLS = ['MonthlyCharges', 'TotalCharges']

def engineer_features(input_path: str = PROCESSED_DATA_PATH, 
                       output_path: str = FEATURES_DATA_PATH,
                       scaler_path: str = SCALER_PATH,
                       is_training: bool = True) -> pd.DataFrame:
    logging.info(f"Engineering features from {input_path}...")
    df = pd.read_csv(input_path)
    
    # 1. Separate Identifiers, Survival Targets, and Covariates
    customer_ids = df['customerID'] if 'customerID' in df.columns else None
    survival_targets = df[['tenure_months', 'churn_event']]
    
    # 2. Binary Encoding for Boolean-like features
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # 3. One-Hot Encoding for Categorical Covariates (drop_first avoids collinearity for CoxPH)
    df_encoded = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True, dtype=int)
    
    # 4. Standardize Continuous Covariates
    if is_training:
        scaler = StandardScaler()
        scaled_nums = scaler.fit_transform(df[NUMERICAL_COLS])
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logging.info(f"Fitted StandardScaler saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        scaled_nums = scaler.transform(df[NUMERICAL_COLS])
        
    df_scaled_nums = pd.DataFrame(scaled_nums, columns=[f"{c}_scaled" for c in NUMERICAL_COLS])
    
    # 5. Concatenate full analytical dataset
    feature_df = pd.concat([survival_targets, df[['SeniorCitizen']], df_scaled_nums, df_encoded], axis=1)
    
    # 6. Save Artifact
    feature_df.to_csv(output_path, index=False)
    logging.info(f"Engineered dataset saved to {output_path} | Shape: {feature_df.shape}")
    return feature_df

if __name__ == "__main__":
    engineer_features()