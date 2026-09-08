import pytest
import os
import joblib
from lifelines import CoxPHFitter
from backend.src.model_training import train_cox_model, run_logrank_analysis, COX_MODEL_PATH

def test_logrank_significance():
    p_val = run_logrank_analysis()
    assert p_val < 0.05, f"Log-Rank p-value ({p_val}) is not statistically significant."

def test_model_training_and_c_index():
    cph = train_cox_model()
    
    assert os.path.exists(COX_MODEL_PATH), "Model artifact was not serialized."
    assert isinstance(cph, CoxPHFitter), "Trained object is not a CoxPHFitter instance."
    
    # Statistical sanity check: C-index must exceed 0.70 for predictive power
    c_index = cph.concordance_index_
    assert c_index >= 0.70, f"Concordance index too low: {c_index:.4f}"
    
    # Check hazard ratios (weights) are non-empty and non-null
    hazard_ratios = cph.hazard_ratios_
    assert not hazard_ratios.isnull().any(), "Hazard ratios contain NaN values."