import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime, timedelta

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Sales Demand Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Sales Forecasting System")
st.markdown("**Predict daily Units Sold** using historical patterns, pricing, and external factors.")

# ========================
# LOAD MODEL & ARTIFACTS
# ========================
@st.cache_resource
def load_model():
    model = joblib.load('../models/best_xgboost_model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    feature_names = joblib.load('../models/feature_names.txt') if 'feature_names.txt' in open('../models/feature_names.txt') else None
    return model, scaler

model, scaler = load_model()

# ========================
# SIDEBAR - USER INPUTS
# ========================
st.sidebar.header("Input Parameters")

inventory_level = st.sidebar.number_input("Inventory Level", min_value=0, value=150)
price = st.sidebar.number_input("Price ($)", min_value=0.0, value=65.0, step=0.1)
discount = st.sidebar.number_input("Discount (%)", min_value=0, max_value=25, value=5)
promotion = st.sidebar.selectbox("Promotion Active?", [0, 1], index=0)
weather = st.sidebar.selectbox("Weather Condition", ["Sunny", "Cloudy", "Snowy", "Rainy"])
seasonality = st.sidebar.selectbox("Seasonality", ["Winter", "Spring", "Summer", "Autumn"])
epidemic = st.sidebar.selectbox("Epidemic Active?", [0, 1], index=0)

# Date input for prediction
prediction_date = st.sidebar.date_input("Prediction Date", datetime.today() + timedelta(days=1))

# ========================
# PREDICTION
# ========================
if st.sidebar.button("🔮 Predict Daily Units Sold", type="primary"):
    # Create input row
    input_data = pd.DataFrame({
        'Inventory Level': [inventory_level],
        'Price': [price],
        'Discount': [discount],
        'Promotion': [promotion],
        'Weather Condition': [weather],
        'Seasonality': [seasonality],
        'Epidemic': [epidemic],
        'Year': [prediction_date.year],
        'Month': [prediction_date.month],
        'DayOfWeek': [prediction_date.weekday()],
        'IsWeekend': [1 if prediction_date.weekday() >= 5 else 0],
        # Add lag and rolling features as 0 for simplicity (can be improved later)
        'Units_Sold_Lag1': [0],
        'Units_Sold_Rolling7': [0],
    })

    # Encode categorical columns
    cat_cols = ['Weather Condition', 'Seasonality']
    for col in cat_cols:
        # Simple encoding (you can improve with saved encoders)
        input_data[col] = pd.Categorical(input_data[col]).codes

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"**Predicted Daily Units Sold: {prediction:.0f} units**")
    st.info(f"📅 Date: {prediction_date.strftime('%Y-%m-%d')}")

    # Simple recommendation
    if prediction > 100:
        st.balloons()
        st.success("🔥 High demand expected - Consider increasing stock!")
    elif prediction < 40:
        st.warning("⚠️ Low demand expected - Be cautious with ordering.")

# ========================
# FOOTER
# ========================
st.caption("Sales Forecasting System | Built with XGBoost | University of Mindanao")