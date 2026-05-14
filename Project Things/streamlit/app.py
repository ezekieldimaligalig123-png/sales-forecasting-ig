import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import sqlite3
import os
import joblib
import gdown

GDRIVE_FILE_ID = "1M6aagrvIUHM6L3fGszl679X3DC31qiE6"
ARTIFACTS_PATH = "model_artifacts/artifacts.pkl"

def download_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        os.makedirs("model_artifacts", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, ARTIFACTS_PATH, quiet=False)

download_artifacts()

@st.cache_resource
def load_artifacts():
    arts = joblib.load('model_artifacts/artifacts.pkl')
    return (
        arts['model'],
        arts['scaler'],
        arts['train_columns'],
        arts['selected_columns'],
    )

try:
    stack_model, scaler, train_columns, selected_cols = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)

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
    conn.commit()
    conn.close()

init_db()


def build_input(prediction_date, inventory_level, price, discount,
                promotion_val, weather, seasonality, epidemic_val):
    dt = pd.Timestamp(prediction_date)

    row = {
        'Inventory Level'   : inventory_level,
        'Price'             : price,
        'Discount'          : discount,
        'Promotion'         : promotion_val,
        'Epidemic'          : epidemic_val,
        'Year'              : dt.year,
        'IsWeekend'         : int(dt.dayofweek >= 5),
        'Store ID_encoded'  : 0.0,
        'Product ID_encoded': 0.0,
        'lag_7'             : 0.0,
        'lag_30'            : 0.0,
        'roll_mean_7'       : 0.0,
        'roll_mean_30'      : 0.0,
        'roll_std_7'        : 0.0,
        'Price_x_Discount'  : price * discount,
        'Effective_Price'   : price * (1 - discount / 100),
        'Inventory_log'     : np.log1p(inventory_level),
        'Month_sin'         : np.sin(2 * np.pi * dt.month / 12),
        'Month_cos'         : np.cos(2 * np.pi * dt.month / 12),
        'DOW_sin'           : np.sin(2 * np.pi * dt.dayofweek / 7),
        'DOW_cos'           : np.cos(2 * np.pi * dt.dayofweek / 7),
    }

    for w in ['Rainy', 'Snowy', 'Sunny']:
        row[f'Weather Condition_{w}'] = int(weather == w)

    for s in ['Spring', 'Summer', 'Winter']:
        row[f'Seasonality_{s}'] = int(seasonality == s)

    for cat in ['Electronics', 'Furniture', 'Groceries', 'Toys']:
        row[f'Category_{cat}'] = 0

    for s in ['Spring', 'Summer', 'Winter']:
        row[f'Promo_x_Seasonality_{s}'] = promotion_val * row[f'Seasonality_{s}']

    df_input = pd.DataFrame([row])
    df_input = df_input.reindex(columns=train_columns, fill_value=0)

    num_cols = df_input.select_dtypes(include=['int64', 'float64', 'bool']).columns
    df_input[num_cols] = scaler.transform(df_input[num_cols])

    df_final = df_input.reindex(columns=selected_cols, fill_value=0)

    return df_final


def predict(prediction_date, inventory_level, price, discount,
            promotion_val, weather, seasonality, epidemic_val):
    X = build_input(prediction_date, inventory_level, price, discount,
                    promotion_val, weather, seasonality, epidemic_val)
    pred = stack_model.predict(X)[0]
    return max(0, int(round(pred)))


st.set_page_config(page_title="Sales Demand Forecaster", layout="wide")

st.title("Sales Forecasting System")
st.markdown("**Daily Units Sold Predictor — Stacked Ensemble (XGBoost + Random Forest)**")

if not MODEL_LOADED:
    st.error(f"Model not found. Run FinalModel.ipynb first to generate model_artifacts/artifacts.pkl. Error: {MODEL_ERROR}")
    st.stop()

st.sidebar.header("Prediction Inputs")

col1, col2 = st.sidebar.columns(2)

with col1:
    inventory_level = st.number_input("Inventory Level", min_value=0, value=120)
    price           = st.number_input("Price ($)", min_value=10.0, value=65.0, step=0.5)
    discount        = st.number_input("Discount (%)", min_value=0, max_value=30, value=8)

with col2:
    promotion   = st.selectbox("Promotion Active?", ["No", "Yes"], index=0)
    weather     = st.selectbox("Weather Condition", ["Cloudy", "Rainy", "Snowy", "Sunny"])
    seasonality = st.selectbox("Season", ["Autumn", "Spring", "Summer", "Winter"])
    epidemic    = st.selectbox("Epidemic / Special Event?", ["No", "Yes"], index=0)

prediction_date = st.sidebar.date_input("Prediction Date", datetime.today() + timedelta(days=1))

predict_button = st.sidebar.button("Predict Units Sold", type="primary", use_container_width=True)
view_history   = st.sidebar.button("View Prediction History", use_container_width=True)

promotion_val = 1 if promotion == "Yes" else 0
epidemic_val  = 1 if epidemic  == "Yes" else 0

if predict_button:
    with st.spinner("Running model..."):
        predicted_units = predict(
            prediction_date, inventory_level, price, discount,
            promotion_val, weather, seasonality, epidemic_val
        )

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

    st.success(f"**Predicted Daily Units Sold: {predicted_units} units**")
    st.info(f"Date: {prediction_date.strftime('%Y-%m-%d')} ({prediction_date.strftime('%A')})")

    if predicted_units >= 110:
        st.balloons()
        st.success("High Demand Expected — Prepare more stock!")
    elif predicted_units >= 70:
        st.info("Moderate Demand")
    else:
        st.warning("Low Demand Expected")

if view_history:
    st.subheader("Prediction History")

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

st.caption("Sales Forecasting System | Stacked Ensemble (XGBoost + Random Forest) | SQLite Database Enabled")
