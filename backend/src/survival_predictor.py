from pathlib import Path
import logging
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================
# survival_predictor.py is located inside:
#
# customer_churn_survival_mlops/
# └── src/
#     └── survival_predictor.py
#
# Therefore Path(__file__).resolve().parent points directly
# to the src/ directory.
# ============================================================

SRC_DIR = Path(__file__).resolve().parent

MODEL_PATH = SRC_DIR / "models" / "cox_ph_model.pkl"
SCALER_PATH = SRC_DIR / "data" / "processed" / "scaler.pkl"
FEATURES_PATH = SRC_DIR / "data" / "processed" / "telco_churn_features.csv"


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

NUMERICAL_COLUMNS = [
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

REQUIRED_INPUT_COLUMNS = (
    ["SeniorCitizen"]
    + NUMERICAL_COLUMNS
    + CATEGORICAL_COLUMNS
)


# ============================================================
# SURVIVAL PREDICTOR
# ============================================================

class SurvivalPredictor:

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        features_path: Path = FEATURES_PATH,
    ):
        """
        Initialize the survival prediction engine.

        Parameters
        ----------
        model_path : Path
            Saved Cox proportional-hazards model.

        scaler_path : Path
            Saved StandardScaler used during feature engineering.

        features_path : Path
            Processed feature dataset used to recover the exact
            training feature schema.
        """

        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.features_path = Path(features_path)

        # ----------------------------------------------------
        # Validate artifacts
        # ----------------------------------------------------

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at:\n{self.model_path}"
            )

        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Scaler artifact not found at:\n{self.scaler_path}"
            )

        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found at:\n{self.features_path}"
            )

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        self.model: CoxPHFitter = joblib.load(self.model_path)

        # ----------------------------------------------------
        # Load scaler
        # ----------------------------------------------------

        self.scaler = joblib.load(self.scaler_path)

        # ----------------------------------------------------
        # Load reference feature schema
        # ----------------------------------------------------

        ref_df = pd.read_csv(
            self.features_path,
            nrows=1
        )

        self.feature_columns = [
            column
            for column in ref_df.columns
            if column not in [
                "tenure_months",
                "churn_event",
            ]
        ]

        logger.info(
            "SurvivalPredictor initialized successfully."
        )

        logger.info(
            "Model path: %s",
            self.model_path
        )

        logger.info(
            "Scaler path: %s",
            self.scaler_path
        )

        logger.info(
            "Expected model features: %d",
            len(self.feature_columns)
        )

        logger.info(
            "Model coefficients: %d",
            len(self.model.params_)
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        model_features = list(self.model.params_.index)

        missing_model_features = [
            column
            for column in model_features
            if column not in self.feature_columns
        ]

        if missing_model_features:
            raise ValueError(
                "Feature schema mismatch.\n"
                f"Features expected by model but missing from "
                f"reference dataset:\n{missing_model_features}"
            )

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    def _validate_input(
        self,
        raw_input: Dict[str, Any]
    ) -> None:
        """
        Validate customer input before preprocessing.
        """

        if not isinstance(raw_input, dict):
            raise TypeError(
                "raw_input must be a dictionary."
            )

        missing_fields = [
            column
            for column in REQUIRED_INPUT_COLUMNS
            if column not in raw_input
        ]

        if missing_fields:
            raise ValueError(
                "Missing required input fields: "
                + ", ".join(missing_fields)
            )

        # ----------------------------------------------------
        # Numerical validation
        # ----------------------------------------------------

        try:
            monthly_charges = float(
                raw_input["MonthlyCharges"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                "MonthlyCharges must be numeric."
            )

        try:
            total_charges = float(
                raw_input["TotalCharges"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                "TotalCharges must be numeric."
            )

        if monthly_charges <= 0:
            raise ValueError(
                "MonthlyCharges must be greater than 0."
            )

        if total_charges < 0:
            raise ValueError(
                "TotalCharges cannot be negative."
            )

        # ----------------------------------------------------
        # Senior citizen validation
        # ----------------------------------------------------

        try:
            senior_citizen = int(
                raw_input["SeniorCitizen"]
            )
        except (TypeError, ValueError):
            raise ValueError(
                "SeniorCitizen must be 0 or 1."
            )

        if senior_citizen not in [0, 1]:
            raise ValueError(
                "SeniorCitizen must be either 0 or 1."
            )

    # ========================================================
    # COMPATIBILITY WRAPPERS
    # ========================================================

    def preprocess_payload(
        self,
        raw_input: Dict[str, Any]
    ) -> pd.DataFrame:
        """Backward-compatible alias used by the prediction tests."""
        return self.preprocess_input(raw_input)

    def _normalize_risk_tier(
        self,
        risk_tier: str,
    ) -> str:
        """Map legacy risk labels to the latest test contract."""
        map_values = {
            "HEALTHY": "STABLE",
            "MODERATE": "ELEVATED",
            "CRITICAL": "CRITICAL",
        }
        return map_values.get(risk_tier, risk_tier)

    # ========================================================
    # PREPROCESS INPUT
    # ========================================================

    def preprocess_input(
        self,
        raw_input: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Convert a raw customer dictionary into the exact
        feature representation expected by the trained
        CoxPH model.

        Parameters
        ----------
        raw_input : dict
            Customer information.

        Returns
        -------
        pandas.DataFrame
            One-row feature matrix aligned to training schema.
        """

        self._validate_input(raw_input)

        # ----------------------------------------------------
        # Create one-row DataFrame
        # ----------------------------------------------------

        df_raw = pd.DataFrame(
            [raw_input]
        )

        # ----------------------------------------------------
        # Numerical features
        # ----------------------------------------------------

        numerical_data = df_raw[
            NUMERICAL_COLUMNS
        ].astype(float)

        scaled_nums = self.scaler.transform(
            numerical_data
        )

        scaled_df = pd.DataFrame(
            scaled_nums,
            columns=[
                f"{column}_scaled"
                for column in NUMERICAL_COLUMNS
            ],
            index=df_raw.index,
        )

        # ----------------------------------------------------
        # SeniorCitizen
        # ----------------------------------------------------

        senior_df = df_raw[
            ["SeniorCitizen"]
        ].astype(int)

        # ----------------------------------------------------
        # Align EXACTLY with training feature schema
        # ----------------------------------------------------
        #
        # Do not call get_dummies on this one-row payload. With
        # drop_first=True, a single row cannot know the categories
        # seen during training, so valid categorical values can be
        # silently encoded as all zeros. Instead, use the persisted
        # training column names as the encoding schema.
        #
        # ----------------------------------------------------

        final_row = pd.DataFrame(
            0,
            index=[0],
            columns=self.feature_columns,
            dtype=float,
        )

        for column in senior_df.columns:
            if column in final_row.columns:
                final_row.loc[0, column] = float(senior_df.iloc[0][column])

        for column in scaled_df.columns:
            if column in final_row.columns:
                final_row.loc[0, column] = float(scaled_df.iloc[0][column])

        for column in CATEGORICAL_COLUMNS:
            value = str(raw_input[column])
            dummy_column = f"{column}_{value}"

            # The omitted reference category intentionally leaves all
            # dummies for this field at zero, matching drop_first=True.
            if dummy_column in final_row.columns:
                final_row.loc[0, dummy_column] = 1.0

        # ----------------------------------------------------
        # Final model feature order
        # ----------------------------------------------------

        final_row = final_row[
            self.feature_columns
        ]

        return final_row

    # ========================================================
    # SURVIVAL CURVE SERIALIZATION
    # ========================================================

    def _serialize_survival_curve(
        self,
        survival_curve: pd.DataFrame,
        current_tenure: int,
    ) -> list:
        """
        Convert lifelines survival DataFrame into a JSON-friendly
        list.

        The returned format is:

        [
            {
                "month": 3,
                "retention": 0.92,
                "churn": 0.08
            },
            ...
        ]
        """

        if survival_curve is None:
            return []

        if survival_curve.empty:
            return []

        survival_column = survival_curve.columns[0]

        records = []

        for index, row in survival_curve.iterrows():

            retention = float(
                row[survival_column]
            )

            retention = max(
                0.0,
                min(1.0, retention)
            )

            records.append(
                {
                    "month": float(index),
                    "retention": retention,
                    "churn": float(
                        1.0 - retention
                    ),
                }
            )

        return records

    # ========================================================
    # TARGET SURVIVAL INTERPOLATION
    # ========================================================

    def _get_retention_at_target_month(
        self,
        survival_curve: pd.DataFrame,
        target_month: float,
    ) -> float:
        """
        Get retention probability at the requested target month.

        If the exact month exists, use it.

        Otherwise linearly interpolate between surrounding
        survival-curve points.

        This is more accurate than simply taking the final
        survival-curve value.
        """

        if survival_curve is None:
            raise ValueError(
                "Survival curve is empty."
            )

        if survival_curve.empty:
            raise ValueError(
                "Survival curve is empty."
            )

        survival_column = survival_curve.columns[0]

        months = np.asarray(
            survival_curve.index,
            dtype=float
        )

        retention_values = np.asarray(
            survival_curve[survival_column],
            dtype=float
        )

        # ----------------------------------------------------
        # Exact target
        # ----------------------------------------------------

        exact_matches = np.where(
            np.isclose(
                months,
                target_month
            )
        )[0]

        if len(exact_matches) > 0:

            retention = retention_values[
                exact_matches[0]
            ]

            return float(
                np.clip(
                    retention,
                    0.0,
                    1.0
                )
            )

        # ----------------------------------------------------
        # Interpolation
        # ----------------------------------------------------

        if target_month <= months.min():

            retention = retention_values[
                0
            ]

        elif target_month >= months.max():

            retention = retention_values[
                -1
            ]

        else:

            retention = float(
                np.interp(
                    target_month,
                    months,
                    retention_values
                )
            )

        return float(
            np.clip(
                retention,
                0.0,
                1.0
            )
        )

    # ========================================================
    # RISK TIER
    # ========================================================

    def _get_risk_tier(
        self,
        partial_hazard: float
    ):
        """
        Determine risk tier from the Cox partial hazard.
        """

        if partial_hazard >= 2.0:

            return (
                "CRITICAL",
                "High churn hazard. Deploy account "
                "retention incentive / dedicated "
                "customer-success outreach."
            )

        elif partial_hazard >= 1.2:

            return (
                "MODERATE",
                "Medium churn hazard. Trigger automated "
                "feature engagement onboarding & email "
                "support sequence."
            )

        else:

            return (
                "HEALTHY",
                "Strong retention stability. Good candidate "
                "for loyalty rewards or annual tier upselling."
            )

    # ========================================================
    # TEST-COMPATIBLE PREDICTION API
    # ========================================================

    def predict_risk(
        self,
        raw_input: Dict[str, Any],
        forward_months: int = 6,
    ) -> Dict[str, Any]:
        """Compatibility wrapper matching the prediction-engine tests."""

        if not isinstance(raw_input, dict):
            raise TypeError("raw_input must be a dictionary.")

        if not isinstance(forward_months, (int, float)):
            raise ValueError("forward_months must be a positive integer.")

        forward_months = int(forward_months)

        if forward_months <= 0:
            raise ValueError("forward_months must be greater than 0.")

        payload = dict(raw_input)
        payload["tenure"] = int(
            raw_input.get("tenure_months", raw_input.get("tenure", 1))
        )
        payload["forecast_horizon"] = forward_months

        result = self.predict_risk_profile(payload)

        risk_tier = self._normalize_risk_tier(result["risk_tier"])

        survival_curve = result.get("survival_curve")
        if isinstance(survival_curve, list):
            survival_curve = {
                str(item.get("month", idx)): float(item.get("retention", 0.0))
                for idx, item in enumerate(survival_curve)
            }
        elif survival_curve is None:
            survival_curve = {}

        normalized = {
            "current_tenure_months": result["current_tenure_months"],
            "forecast_horizon_months": result["forecast_horizon_months"],
            "target_horizon_month": result["target_month"],
            "partial_hazard_ratio": result["hazard_ratio_multiplier"],
            "projected_churn_probability": result["projected_churn_pct"],
            "projected_retention_probability": result["projected_retention_pct"],
            "risk_tier": risk_tier,
            "recommended_action": result["recommended_action"],
            "survival_curve": survival_curve,
        }

        return normalized

    # ========================================================
    # MAIN PREDICTION METHOD
    # ========================================================

    def predict_risk_profile(
        self,
        raw_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a complete customer churn survival profile.

        Parameters
        ----------
        raw_input : dict
            Customer profile.

            Required fields include:

            tenure
            forecast_horizon
            gender
            SeniorCitizen
            Partner
            Dependents
            PhoneService
            MultipleLines
            InternetService
            OnlineSecurity
            OnlineBackup
            DeviceProtection
            TechSupport
            StreamingTV
            StreamingMovies
            Contract
            PaperlessBilling
            PaymentMethod
            MonthlyCharges
            TotalCharges

        Returns
        -------
        dict
            Complete risk profile.
        """

        if not isinstance(
            raw_input,
            dict
        ):
            raise TypeError(
                "raw_input must be a dictionary."
            )

        # ----------------------------------------------------
        # Current tenure
        # ----------------------------------------------------

        try:
            current_tenure = int(
                raw_input.get(
                    "tenure",
                    1
                )
            )
        except (TypeError, ValueError):

            raise ValueError(
                "tenure must be an integer."
            )

        if current_tenure < 1:

            current_tenure = 1

        # ----------------------------------------------------
        # Forecast horizon
        # ----------------------------------------------------

        try:
            forecast_horizon = int(
                raw_input.get(
                    "forecast_horizon",
                    6
                )
            )
        except (TypeError, ValueError):

            raise ValueError(
                "forecast_horizon must be an integer."
            )

        if forecast_horizon < 1:

            forecast_horizon = 1

        # ----------------------------------------------------
        # Target month
        # ----------------------------------------------------

        target_month = (
            current_tenure
            + forecast_horizon
        )

        # ----------------------------------------------------
        # Preprocess customer
        # ----------------------------------------------------

        processed_features = (
            self.preprocess_input(
                raw_input
            )
        )

        # ----------------------------------------------------
        # Safety check against model
        # ----------------------------------------------------

        model_features = list(
            self.model.params_.index
        )

        missing_features = [
            column
            for column in model_features
            if column not in processed_features.columns
        ]

        if missing_features:

            raise ValueError(
                "Processed input is missing model features: "
                + ", ".join(missing_features)
            )

        # Ensure exact model order
        processed_features = processed_features[
            model_features
        ]

        # ----------------------------------------------------
        # 1. PARTIAL HAZARD
        # ----------------------------------------------------
        #
        # This is NOT a churn percentage.
        #
        # Example:
        #
        # 6.443x
        #
        # means approximately 6.443 times the reference hazard.
        # ----------------------------------------------------

        partial_hazard = float(
            self.model
            .predict_partial_hazard(
                processed_features
            )
            .iloc[0]
        )

        # ----------------------------------------------------
        # 2. CONDITIONAL SURVIVAL
        # ----------------------------------------------------
        #
        # S(t | T > current_tenure)
        #
        # This answers:
        #
        # "Given that this customer has survived until their
        # current tenure, what is their probability of surviving
        # to future month t?"
        # ----------------------------------------------------

        survival_curve_df = (
            self.model.predict_survival_function(
                processed_features,
                conditional_after=[
                    current_tenure
                ]
            )
        )

        # ----------------------------------------------------
        # Target retention
        # ----------------------------------------------------

        retention_prob = (
            self._get_retention_at_target_month(
                survival_curve_df,
                target_month
            )
        )

        # ----------------------------------------------------
        # Target churn probability
        # ----------------------------------------------------

        churn_prob = (
            1.0
            - retention_prob
        )

        churn_prob = float(
            np.clip(
                churn_prob,
                0.0,
                1.0
            )
        )

        retention_prob = float(
            np.clip(
                retention_prob,
                0.0,
                1.0
            )
        )

        # ----------------------------------------------------
        # 3. RISK TIER
        # ----------------------------------------------------

        risk_tier, recommended_action = (
            self._get_risk_tier(
                partial_hazard
            )
        )

        # ----------------------------------------------------
        # 4. JSON-FRIENDLY SURVIVAL CURVE
        # ----------------------------------------------------

        survival_curve = (
            self._serialize_survival_curve(
                survival_curve_df,
                current_tenure
            )
        )

        # ----------------------------------------------------
        # 5. FINAL RESPONSE
        # ----------------------------------------------------

        result = {

            "current_tenure_months": (
                current_tenure
            ),

            "forecast_horizon_months": (
                forecast_horizon
            ),

            "target_month": (
                target_month
            ),

            "hazard_ratio_multiplier": round(
                partial_hazard,
                3
            ),

            "projected_churn_pct": round(
                churn_prob * 100,
                2
            ),

            "projected_retention_pct": round(
                retention_prob * 100,
                2
            ),

            "risk_tier": (
                risk_tier
            ),

            "recommended_action": (
                recommended_action
            ),

            "survival_curve": (
                survival_curve
            ),
        }

        logger.info(
            "Prediction generated | "
            "Tenure=%s | "
            "Target=%s | "
            "Hazard=%.3fx | "
            "Churn=%.2f%% | "
            "Risk=%s",
            current_tenure,
            target_month,
            partial_hazard,
            churn_prob * 100,
            risk_tier,
        )

        return result


# ============================================================
# OPTIONAL DIRECT TEST
# ============================================================

if __name__ == "__main__":

    predictor = SurvivalPredictor()

    sample_customer = {

        "tenure": 3,

        "forecast_horizon": 6,

        "gender": "Male",

        "SeniorCitizen": 0,

        "Partner": "No",

        "Dependents": "No",

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "No",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "No",

        "StreamingMovies": "No",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 70.0,

        "TotalCharges": 210.0,
    }

    result = predictor.predict_risk_profile(
        sample_customer
    )

    print("\n" + "=" * 60)
    print("CUSTOMER CHURN SURVIVAL PREDICTION")
    print("=" * 60)

    print(
        f"Current Tenure: "
        f"{result['current_tenure_months']} months"
    )

    print(
        f"Forecast Horizon: "
        f"+{result['forecast_horizon_months']} months"
    )

    print(
        f"Target Month: "
        f"{result['target_month']}"
    )

    print(
        f"Hazard Multiplier: "
        f"{result['hazard_ratio_multiplier']}x"
    )

    print(
        f"Projected Churn: "
        f"{result['projected_churn_pct']}%"
    )

    print(
        f"Projected Retention: "
        f"{result['projected_retention_pct']}%"
    )

    print(
        f"Risk Tier: "
        f"{result['risk_tier']}"
    )

    print(
        f"Recommended Action: "
        f"{result['recommended_action']}"
    )

    print(
        f"Survival Curve Points: "
        f"{len(result['survival_curve'])}"
    )

    print("=" * 60)
