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
st.sidebar.header("🔧 Prediction Inputs")

col1, col2 = st.sidebar.columns(2)

with col1:
    inventory_level = st.number_input("Inventory Level", min_value=0, value=120)
    price = st.number_input("Price ($)", min_value=10.0, value=65.0, step=0.5)
    discount = st.number_input("Discount (%)", min_value=0, max_value=30, value=8)

with col2:
    promotion = st.selectbox("Promotion Active?", ["No", "Yes"], index=0)
    weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Rainy", "Snowy"])
    seasonality = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
    epidemic = st.selectbox("Epidemic / Special Event?", ["No", "Yes"], index=0)

prediction_date = st.sidebar.date_input("Prediction Date", datetime.today() + timedelta(days=1))

# Option to view history
view_history = st.sidebar.checkbox("📜 View Prediction History", value=False)

# ========================
# CONVERT YES/NO TO 0/1 FOR MODEL
# ========================
promotion_val = 1 if promotion == "Yes" else 0
epidemic_val = 1 if epidemic == "Yes" else 0

# ========================
# PREDICTION LOGIC (Mock for now)
# ========================
if st.sidebar.button("🔮 Predict Units Sold", type="primary", use_container_width=True):
    
    base_units = 80
    
    # Adjustments based on inputs
    if promotion_val == 1:
        base_units += 35
    if discount >= 10:
        base_units += 20
    if weather in ["Rainy", "Snowy"]:
        base_units -= 15
    if epidemic_val == 1:
        base_units -= 25
    if seasonality in ["Summer", "Winter"]:
        base_units += 10
    
    predicted_units = int(base_units + np.random.normal(0, 8))
    predicted_units = max(20, predicted_units)

    st.success(f"**Predicted Daily Units Sold: {predicted_units} units**")
    st.info(f"📅 **Date**: {prediction_date.strftime('%Y-%m-%d')} | {prediction_date.strftime('%A')}")

    if predicted_units >= 110:
        st.balloons()
        st.success("🚀 High Demand Expected - Prepare more stock!")
    elif predicted_units >= 70:
        st.info("✅ Moderate Demand")
    else:
        st.warning("⚠️ Low Demand Expected")

# ========================
# PREDICTION HISTORY (Sample Data)
# ========================
if view_history:
    st.subheader("📜 Prediction History (Sample)")
    
    sample_history = pd.DataFrame({
        "Date": ["2026-05-10", "2026-05-09", "2026-05-08", "2026-05-07"],
        "Predicted Units": [98, 145, 67, 112],
        "Promotion": ["Yes", "No", "Yes", "No"],
        "Discount (%)": [10, 5, 15, 0],
        "Weather": ["Sunny", "Rainy", "Cloudy", "Sunny"],
        "Epidemic": ["No", "No", "Yes", "No"]
    })
    
    st.dataframe(sample_history, use_container_width=True, hide_index=True)
    
    st.caption("Note: This is sample data. Real history will be saved once database is connected.")

# ========================
# FOOTER
# ========================
st.caption("Sales Forecasting System | XGBoost Model | Sample Version")
