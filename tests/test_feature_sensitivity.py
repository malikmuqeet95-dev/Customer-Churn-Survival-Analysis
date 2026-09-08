import pytest

from backend.src.survival_predictor import SurvivalPredictor


@pytest.fixture
def sample_payload():
    return {
        "tenure": 12,
        "forecast_horizon": 12,
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }


def test_displayed_categorical_features_change_prediction(sample_payload):
    predictor = SurvivalPredictor()
    baseline_features = predictor.preprocess_input(sample_payload)

    changed_payload = sample_payload.copy()
    changed_payload["Contract"] = "Two year"
    changed_payload["InternetService"] = "Fiber optic"
    changed_payload["PaymentMethod"] = "Electronic check"
    changed_features = predictor.preprocess_input(changed_payload)

    assert not baseline_features.equals(changed_features)

    baseline_output = predictor.predict_risk_profile(sample_payload)
    changed_output = predictor.predict_risk_profile(changed_payload)

    assert baseline_output["hazard_ratio_multiplier"] != changed_output[
        "hazard_ratio_multiplier"
    ]
    assert baseline_output["projected_churn_pct"] != changed_output[
        "projected_churn_pct"
    ]
