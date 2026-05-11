import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(
    page_title="Sales Demand Forecaster",
    page_icon="📈",
    layout="wide"
)

st.title("📊 Sales Forecasting System")
st.markdown("**Daily Units Sold Predictor**")

# ========================
# SIDEBAR INPUTS
# ========================
st.sidebar.header("🔧 Input Parameters for Prediction")

col1, col2 = st.sidebar.columns(2)

with col1:
    inventory_level = st.number_input("Inventory Level", min_value=0, value=120)
    price = st.number_input("Price ($)", min_value=10.0, value=65.0, step=0.5)
    discount = st.number_input("Discount (%)", min_value=0, max_value=30, value=8)

with col2:
    promotion = st.selectbox("Promotion Active?", [0, 1], index=0)
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy"])
    seasonality = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    epidemic = st.selectbox("Epidemic / Special Event?", [0, 1], index=0)

prediction_date = st.sidebar.date_input("Prediction Date", datetime.today() + timedelta(days=1))

# ========================
# PREDICTION LOGIC (Mock + Simple Rule-based for now)
# ========================
if st.sidebar.button("🔮 Predict Units Sold", type="primary", use_container_width=True):
    
    # Simple rule-based prediction (replace later with real model)
    base_units = 80
    
    # Adjustments
    if promotion == 1:
        base_units += 35
    if discount >= 10:
        base_units += 20
    if weather in ["Rainy", "Snowy"]:
        base_units -= 15
    if epidemic == 1:
        base_units -= 25
    if seasonality in ["Summer", "Winter"]:
        base_units += 10
    
    # Add some randomness to simulate model behavior
    predicted_units = int(base_units + np.random.normal(0, 8))
    predicted_units = max(20, predicted_units)   # Minimum realistic sales

    # Display Results
    st.success(f"**Predicted Daily Units Sold: {predicted_units} units**")
    
    st.info(f"📅 **Date**: {prediction_date.strftime('%Y-%m-%d')} | {prediction_date.strftime('%A')}")
    
    # Recommendation
    if predicted_units >= 110:
        st.balloons()
        st.success("🚀 **High Demand Expected** - Prepare more stock!")
    elif predicted_units >= 70:
        st.info("✅ **Moderate Demand** - Normal stock level is fine.")
    else:
        st.warning("⚠️ **Low Demand Expected** - Consider reducing order.")

    # Show input summary
    st.subheader("Input Summary")
    input_df = pd.DataFrame({
        "Parameter": ["Inventory", "Price", "Discount", "Promotion", "Weather", "Season", "Epidemic"],
        "Value": [inventory_level, f"${price}", f"{discount}%", promotion, weather, seasonality, epidemic]
    })
    st.table(input_df)

# ========================
# FOOTER
# ========================
st.caption("Sales Forecasting System | XGBoost Model | Sample Version")