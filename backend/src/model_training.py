import os
import pandas as pd
import numpy as np
import joblib
import logging
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

FEATURES_PATH = "data/processed/telco_churn_features.csv"
CLEAN_DATA_PATH = "data/processed/telco_churn_clean.csv"
MODEL_DIR = "models"
COX_MODEL_PATH = os.path.join(MODEL_DIR, "cox_ph_model.pkl")

def run_logrank_analysis(clean_data_path: str = CLEAN_DATA_PATH):
    """Performs statistical hypothesis testing across contract cohorts."""
    logging.info("Running Log-Rank significance tests...")
    df = pd.read_csv(clean_data_path)
    
    m2m = df[df['Contract'] == 'Month-to-month']
    two_yr = df[df['Contract'] == 'Two year']
    
    results = logrank_test(
        durations_A=m2m['tenure_months'], 
        durations_B=two_yr['tenure_months'],
        event_observed_A=m2m['churn_event'],
        event_observed_B=two_yr['churn_event']
    )
    
    logging.info(f"Log-Rank Test (Month-to-month vs Two-year) p-value: {results.p_value:.4e}")
    return results.p_value

def train_cox_model(features_path: str = FEATURES_PATH, 
                    save_path: str = COX_MODEL_PATH, 
                    penalizer: float = 0.05) -> CoxPHFitter:
    """Trains a penalized Cox Proportional Hazards model and logs C-index."""
    logging.info(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    # Train/Validation Split (80/20) based on index shuffling
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Initialize CoxPH with L2 penalization for stability
    cph = CoxPHFitter(penalizer=penalizer)
    
    logging.info(f"Fitting Cox Proportional Hazards Model (Penalizer={penalizer})...")
    cph.fit(train_df, duration_col='tenure_months', event_col='churn_event')
    
    # Evaluate Harrell's Concordance Index on Holdout Validation Set
    val_c_index = cph.score(val_df, scoring_method="concordance_index")
    logging.info(f"Validation Concordance Index (C-Index): {val_c_index:.4f}")
    
    # Persist model artifact
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(cph, save_path)
    logging.info(f"Trained CoxPH model saved successfully to {save_path}")
    
    return cph

if __name__ == "__main__":
    run_logrank_analysis()
    train_cox_model()