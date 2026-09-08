import pytest

from backend.src.survival_predictor import SurvivalPredictor


# ============================================================
# FIXTURE: SURVIVAL PREDICTOR
# ============================================================

@pytest.fixture
def predictor():
    return SurvivalPredictor()


# ============================================================
# FIXTURE: SAMPLE CUSTOMER
# ============================================================

@pytest.fixture
def sample_payload():
    return {
        'gender': 'Male',
        'SeniorCitizen': 1,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure_months': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'Yes',
        'InternetService': 'DSL',
        'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport': 'Yes',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'One year',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)',
        'MonthlyCharges': 55.0,
        'TotalCharges': 660.0
    }


# ============================================================
# TEST 1: PREPROCESSING
# ============================================================

def test_preprocessing_shape(predictor, sample_payload):

    features = predictor.preprocess_payload(
        sample_payload
    )

    # One customer = one row
    assert features.shape[0] == 1

    # Number of columns must exactly match training schema
    assert features.shape[1] == len(
        predictor.feature_columns
    )

    # No missing values
    assert not features.isnull().any().any()

    # All model features must be numeric
    assert features.select_dtypes(
        exclude=['number']
    ).empty


# ============================================================
# TEST 2: PREDICTION OUTPUT STRUCTURE
# ============================================================

def test_prediction_output_structure(
    predictor,
    sample_payload
):

    output = predictor.predict_risk(
        sample_payload,
        forward_months=12
    )

    # --------------------------------------------------------
    # Required output fields
    # --------------------------------------------------------

    assert "current_tenure_months" in output
    assert "forecast_horizon_months" in output
    assert "target_horizon_month" in output
    assert "partial_hazard_ratio" in output
    assert "projected_churn_probability" in output
    assert "projected_retention_probability" in output
    assert "risk_tier" in output
    assert "recommended_action" in output
    assert "survival_curve" in output

    # --------------------------------------------------------
    # Basic value validation
    # --------------------------------------------------------

    assert output["current_tenure_months"] == 12

    assert output["forecast_horizon_months"] == 12

    assert output["target_horizon_month"] == 24

    assert output["partial_hazard_ratio"] > 0

    # --------------------------------------------------------
    # Probability validation
    # --------------------------------------------------------

    assert 0.0 <= output[
        "projected_retention_probability"
    ] <= 100.0

    assert 0.0 <= output[
        "projected_churn_probability"
    ] <= 100.0

    # Retention + churn should approximately equal 100%
    assert (
        abs(
            (
                output["projected_retention_probability"]
                + output["projected_churn_probability"]
            ) - 100.0
        ) < 0.1
    )

    # --------------------------------------------------------
    # Risk tier validation
    # --------------------------------------------------------

    assert output["risk_tier"] in [
        "STABLE",
        "ELEVATED",
        "CRITICAL"
    ]

    # --------------------------------------------------------
    # Business action must exist
    # --------------------------------------------------------

    assert isinstance(
        output["recommended_action"],
        str
    )

    assert len(
        output["recommended_action"]
    ) > 0

    # --------------------------------------------------------
    # Complete survival curve must exist
    # --------------------------------------------------------

    assert isinstance(
        output["survival_curve"],
        dict
    )

    assert len(
        output["survival_curve"]
    ) > 0


def test_app_handles_list_of_dict_survival_curve():
    import app

    raw_curve = [
        {"month": 0.0, "retention": 1.0},
        {"month": 1.0, "retention": 0.93},
        {"month": 2.0, "retention": 0.80},
    ]

    x_timeline, y_values = app.normalize_survival_curve(raw_curve)

    assert x_timeline == [0.0, 1.0, 2.0]
    assert y_values == [100.0, 93.0, 80.0]


# ============================================================
# TEST 3: FORECAST HORIZON
# ============================================================

def test_different_forecast_horizon(
    predictor,
    sample_payload
):

    output = predictor.predict_risk(
        sample_payload,
        forward_months=6
    )

    assert output[
        "forecast_horizon_months"
    ] == 6

    assert output[
        "target_horizon_month"
    ] == 18

    assert 0.0 <= output[
        "projected_retention_probability"
    ] <= 100.0

    assert 0.0 <= output[
        "projected_churn_probability"
    ] <= 100.0


# ============================================================
# TEST 4: INVALID FORECAST HORIZON
# ============================================================

def test_invalid_forecast_horizon(
    predictor,
    sample_payload
):

    with pytest.raises(ValueError):

        predictor.predict_risk(
            sample_payload,
            forward_months=0
        )


# ============================================================
# TEST 5: MISSING REQUIRED INPUT
# ============================================================

def test_missing_required_input(
    predictor,
    sample_payload
):

    invalid_payload = sample_payload.copy()

    del invalid_payload["MonthlyCharges"]

    with pytest.raises(ValueError):

        predictor.predict_risk(
            invalid_payload,
            forward_months=12
        )

