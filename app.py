import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backend.src.survival_predictor import SurvivalPredictor

st.set_page_config(page_title="Churn Survival Intelligence", layout="wide")

st.title("⚡ ChurnGuard: Customer Retention Survival AI")
st.markdown("Dynamic Time-to-Event Survival Forecasting Engine")

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
    curve_df = result['survival_curve']
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve_df.index, 
        y=curve_df.values.flatten() * 100,
        mode='lines+markers',
        name='Survival Probability',
        line=dict(color='#0284c7', width=3)
    ))
    fig.update_layout(
        title=f"Survival Trajectory (Active: {tenure} Mo | Horizon: +{forecast_horizon} Mo)",
        xaxis_title="Tenure Timeline (Months)",
        yaxis_title="Retention Probability (%)",
        yaxis_range=[0, 105],
        template="plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)