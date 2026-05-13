import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sqlite3
import os

# ========================
# DATABASE SETUP
# ========================
DB_PATH = "sales_predictions.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT,
            predicted_units INTEGER,
            inventory_level INTEGER,
            price REAL,
            discount INTEGER,
            promotion INTEGER,
            weather TEXT,
            seasonality TEXT,
            epidemic INTEGER,
            timestamp TEXT
        )
    ''')
    conn.close()

# Initialize database
init_db()

# ========================
# PAGE CONFIG
# ========================
st.set_page_config(page_title="Sales Demand Forecaster", page_icon="📈", layout="wide")

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

predict_button = st.sidebar.button("🔮 Predict Units Sold", type="primary", use_container_width=True)
view_history = st.sidebar.button("📜 View Prediction History", use_container_width=True)

# Convert Yes/No to 0/1
promotion_val = 1 if promotion == "Yes" else 0
epidemic_val = 1 if epidemic == "Yes" else 0

# ========================
# MAKE PREDICTION + SAVE TO DB
# ========================
if predict_button:
    # Simple mock prediction logic (replace with real model later)
    base_units = 80
    if promotion_val == 1: base_units += 35
    if discount >= 10: base_units += 20
    if weather in ["Rainy", "Snowy"]: base_units -= 15
    if epidemic_val == 1: base_units -= 25
    if seasonality in ["Summer", "Winter"]: base_units += 10
    
    predicted_units = int(base_units + np.random.normal(0, 8))
    predicted_units = max(20, predicted_units)

    # Save to database
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO predictions 
        (prediction_date, predicted_units, inventory_level, price, discount, 
         promotion, weather, seasonality, epidemic, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        prediction_date.strftime('%Y-%m-%d'),
        predicted_units,
        inventory_level,
        price,
        discount,
        promotion_val,
        weather,
        seasonality,
        epidemic_val,
        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

    # Display result
    st.success(f"**Predicted Daily Units Sold: {predicted_units} units**")
    st.info(f"📅 **Date**: {prediction_date.strftime('%Y-%m-%d')} ({prediction_date.strftime('%A')})")

    if predicted_units >= 110:
        st.balloons()
        st.success("🚀 High Demand Expected - Prepare more stock!")
    elif predicted_units >= 70:
        st.info("✅ Moderate Demand")
    else:
        st.warning("⚠️ Low Demand Expected")

# ========================
# VIEW PREDICTION HISTORY
# ========================
if view_history:
    st.subheader("📜 Prediction History")
    
    conn = sqlite3.connect(DB_PATH)
    df_history = pd.read_sql_query("""
        SELECT 
            prediction_date as Date,
            predicted_units as "Predicted Units",
            inventory_level as "Inventory Level",
            price as "Price ($)",
            discount as "Discount (%)",
            CASE WHEN promotion = 1 THEN 'Yes' ELSE 'No' END as Promotion,
            weather as Weather,
            seasonality as Season,
            CASE WHEN epidemic = 1 THEN 'Yes' ELSE 'No' END as Epidemic,
            timestamp as "Predicted At"
        FROM predictions 
        ORDER BY id DESC 
        LIMIT 20
    """, conn)
    conn.close()

    if df_history.empty:
        st.info("No predictions yet. Make your first prediction!")
    else:
        st.dataframe(df_history, use_container_width=True, hide_index=True)

# ========================
# FOOTER
# ========================
st.caption("Sales Forecasting System | XGBoost Model | SQLite Database Enabled")
