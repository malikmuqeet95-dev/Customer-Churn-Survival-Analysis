import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.src.survival_predictor import SurvivalPredictor

st.set_page_config(page_title="Churn Survival Intelligence", layout="wide")

st.title("⚡ ChurnGuard: Customer Retention Survival AI")
st.markdown("Dynamic Time-to-Event Survival Forecasting Engine")


def normalize_survival_curve(raw_curve):
    """Convert survival curve payloads into x/y arrays for Plotly."""
    if isinstance(raw_curve, pd.DataFrame):
        x_timeline = [float(idx) for idx in raw_curve.index]
        y_values = [
            float(val) * 100.0 if float(val) <= 1.0 else float(val)
            for val in raw_curve.iloc[:, 0].values
        ]
        return x_timeline, y_values

    if isinstance(raw_curve, dict):
        if 'timeline' in raw_curve and 'survival_prob' in raw_curve:
            x_timeline = [float(t) for t in raw_curve['timeline']]
            y_values = [
                float(p) * 100.0 if float(p) <= 1.0 else float(p)
                for p in raw_curve['survival_prob']
            ]
            return x_timeline, y_values

        if 'survival_curve' in raw_curve and isinstance(raw_curve['survival_curve'], list):
            return normalize_survival_curve(raw_curve['survival_curve'])

    if isinstance(raw_curve, list) and raw_curve and isinstance(raw_curve[0], dict):
        x_timeline = []
        y_values = []

        for item in raw_curve:
            if not isinstance(item, dict):
                continue

            month = item.get('month', item.get('time'))
            retention = item.get('retention', item.get('survival_prob'))

            if month is None or retention is None:
                continue

            x_timeline.append(float(month))
            value = float(retention)
            y_values.append(value * 100.0 if value <= 1.0 else value)

        return x_timeline, y_values

    raw_series = pd.Series(raw_curve)
    x_timeline = [float(idx) for idx in raw_series.index]
    y_values = [
        float(val) * 100.0 if float(val) <= 1.0 else float(val)
        for val in raw_series.values
    ]
    return x_timeline, y_values


predictor = SurvivalPredictor()

# Sidebar / Input Form
with st.sidebar:
    st.header("Customer Profile")
    tenure = st.slider("Active Tenure (Months)", 1, 72, 6)
    forecast_horizon = st.slider("Forecast Horizon (+Months)", 1, 24, 6)
    monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 150.0, 75.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0, 450.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

payload = {
    'tenure': tenure,
    'forecast_horizon': forecast_horizon,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'Contract': contract,
    'InternetService': internet_service,
    'TechSupport': tech_support,
    'PaymentMethod': payment_method,
    'PaperlessBilling': 'Yes',
    'SeniorCitizen': 0,
    'gender': 'Male',
    'Partner': 'No',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No'
}

if st.button("Calculate Survival Forecast", type="primary"):
    result = predictor.predict_risk_profile(payload)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Hazard Multiplier", f"{result['hazard_ratio_multiplier']}x")
    col2.metric("Risk Tier", result['risk_tier'])
    col3.metric("Projected Retention", f"{result['projected_retention_pct']}%")
    
    st.info(f"**Recommended Action:** {result['recommended_action']}")
    
    # Plotly Survival Curve
    raw_curve = result['survival_curve']
    x_timeline, y_values = normalize_survival_curve(raw_curve)

    # 2. Render Plotly Curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_timeline, 
        y=y_values,
        mode='lines+markers',
        name='Survival Probability',
        line=dict(color='#0284c7', width=3),
        marker=dict(size=4, color='#38bdf8')
    ))
    fig.update_layout(
        title=f"Survival Trajectory (Active: {tenure} Mo | Horizon: +{forecast_horizon} Mo)",
        xaxis_title="Tenure Timeline (Months)",
        yaxis_title="Retention Probability (%)",
        yaxis_range=[0, 105],
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)