from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.src.survival_predictor import SurvivalPredictor


# ============================================================
# PATHS
# ============================================================

# backend/main.py is located inside:
# project_root/
# └── backend/
#     └── main.py
#
# So the project root is simply the parent of backend/, not
# the grandparent of backend/.

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Customer Churn Survival API",
    description="FastAPI backend serving customer churn survival predictions.",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

CSS_DIR = FRONTEND_DIR / "css"
JS_DIR = FRONTEND_DIR / "js"

if CSS_DIR.exists():
    app.mount(
        "/css",
        StaticFiles(directory=str(CSS_DIR)),
        name="css",
    )

if JS_DIR.exists():
    app.mount(
        "/js",
        StaticFiles(directory=str(JS_DIR)),
        name="js",
    )


# ============================================================
# INITIALIZE SURVIVAL PREDICTOR
# ============================================================

try:
    predictor = SurvivalPredictor()

except Exception as e:
    predictor = None

    print(
        "WARNING: SurvivalPredictor could not be initialized."
    )

    print(
        f"Reason: {e}"
    )


# ============================================================
# REQUEST MODEL
# ============================================================

class CustomerPayload(BaseModel):

    tenure: int = Field(
        default=3,
        ge=1,
        le=72,
    )

    forecast_horizon: int = Field(
        default=6,
        ge=1,
        le=36,
    )

    MonthlyCharges: float = Field(
        default=75.0,
        ge=10.0,
        le=250.0,
    )

    TotalCharges: float = Field(
        default=225.0,
        ge=0.0,
    )

    SeniorCitizen: int = Field(
        default=0,
        ge=0,
        le=1,
    )

    gender: str = Field(
        default="Male"
    )

    Partner: str = Field(
        default="No"
    )

    Dependents: str = Field(
        default="No"
    )

    PhoneService: str = Field(
        default="Yes"
    )

    MultipleLines: str = Field(
        default="No"
    )

    InternetService: str = Field(
        default="Fiber optic"
    )

    OnlineSecurity: str = Field(
        default="No"
    )

    OnlineBackup: str = Field(
        default="No"
    )

    DeviceProtection: str = Field(
        default="No"
    )

    TechSupport: str = Field(
        default="No"
    )

    StreamingTV: str = Field(
        default="No"
    )

    StreamingMovies: str = Field(
        default="No"
    )

    Contract: str = Field(
        default="Month-to-month"
    )

    PaperlessBilling: str = Field(
        default="Yes"
    )

    PaymentMethod: str = Field(
        default="Electronic check"
    )


# ============================================================
# ROOT / FRONTEND
# ============================================================

@app.get("/")
def serve_index():

    index_file = FRONTEND_DIR / "index.html"

    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "index.html not found at: "
                f"{index_file}"
            ),
        )

    return FileResponse(
        str(index_file)
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "predictor_loaded": predictor is not None,
    }


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/api/predict")
def predict_churn(
    payload: CustomerPayload
):

    # --------------------------------------------------------
    # Check predictor
    # --------------------------------------------------------

    if predictor is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Survival prediction model is not "
                "available."
            ),
        )

    try:

        # ----------------------------------------------------
        # Convert Pydantic model to dictionary
        # ----------------------------------------------------

        data_dict = payload.model_dump()

        # ----------------------------------------------------
        # Run prediction
        # ----------------------------------------------------

        result = predictor.predict_risk_profile(
            data_dict
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # survival_predictor.py returns:
        #
        # [
        #     {
        #         "month": 3,
        #         "retention": 0.92,
        #         "churn": 0.08
        #     },
        #     ...
        # ]
        #
        # Therefore we DO NOT use .index or .values.
        # ----------------------------------------------------

        curve = result.get(
            "survival_curve",
            []
        )

        timeline = []
        survival_prob = []
        churn_prob = []

        for point in curve:

            timeline.append(
                float(point["month"])
            )

            survival_prob.append(
                round(
                    float(point["retention"]),
                    4
                )
            )

            churn_prob.append(
                round(
                    float(point["churn"]),
                    4
                )
            )

        # ----------------------------------------------------
        # Return API response
        # ----------------------------------------------------

        return {

            "status": "success",

            "current_tenure": (
                result["current_tenure_months"]
            ),

            "forecast_horizon": (
                result["forecast_horizon_months"]
            ),

            "target_month": (
                result["target_month"]
            ),

            "hazard_ratio": (
                result["hazard_ratio_multiplier"]
            ),

            "projected_churn": (
                result["projected_churn_pct"]
            ),

            "projected_retention": (
                result["projected_retention_pct"]
            ),

            "risk_tier": (
                result["risk_tier"]
            ),

            "recommended_action": (
                result["recommended_action"]
            ),

            "forecast_curve": {

                "timeline": timeline,

                "survival_prob": survival_prob,

                "churn_prob": churn_prob,

            },
        }

    # --------------------------------------------------------
    # Expected prediction/input errors
    # --------------------------------------------------------

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    # --------------------------------------------------------
    # Unexpected server errors
    # --------------------------------------------------------

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(e)}"
            ),
        )