"""
WMS Forecast API Module
-----------------------
FastAPI implementation for real-time demand forecasting.
Loads trained XGBoost artifacts, calculates dynamic features using 
historical SQLite context, and generates visual forecast plots.
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Importar as utilidades de features do projeto
from src.utils.features import (
    create_features, 
    add_weekday_weekend_flags, 
    add_holidays, 
    add_calendar_events,
    add_lags,
    add_diff,
    add_ewma
)

app = FastAPI(title="WMS Forecast API")

# ==============================================================================
# ARTIFACT LOADING
# ==============================================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(ROOT_DIR, 'models/xgboost/xgboost_final_model.joblib')
PREPROCESSOR_PATH = os.path.join(ROOT_DIR, 'models/xgboost/preprocessor.joblib')
DB_PATH = os.path.join(ROOT_DIR, 'data/api_forecast.db')

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    raise RuntimeError("Model artifacts not found. Please train the model first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# ==============================================================================
# SCHEMA & DATA UTILITIES
# ==============================================================================
class PredictRequest(BaseModel):
    brand: str
    hierarchy: str
    date: str  # Formato YYYY-MM-DD

def get_historical_context(brand, hier, date):
    """Retrieves the last 60 days of history to compute lags and EWMAs."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DATE, BRAND, PRODUCTHIERARCHY3, QUANTITY 
        FROM history 
        WHERE BRAND = ? AND PRODUCTHIERARCHY3 = ? AND DATE < ?
        ORDER BY DATE DESC LIMIT 60
    """
    df = pd.read_sql(query, conn, params=(brand, hier, date))
    conn.close()
    return df.sort_values('DATE')

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================
def save_prediction_plot(brand, hierarchy, date, predicted_qty, history_df):
    """Generates a PNG plot comparing history vs forecast."""
    plt.figure(figsize=(10, 5))
    
    # 1. Format Historical Data
    history_df = history_df.copy()
    history_df['DATE_DT'] = pd.to_datetime(history_df['DATE'])
    history_df = history_df.sort_values('DATE_DT')
    
    # 2. Plot Real Values
    plt.plot(history_df['DATE_DT'], history_df['QUANTITY'], 
             label='History', marker='o', linestyle='-', alpha=0.6)
    
    # 3. Plot Forecasted Value
    forecast_dt = pd.to_datetime(date)
    plt.scatter(forecast_dt, predicted_qty, color='red', s=120, 
                label=f'Forecast: {predicted_qty}', zorder=5)
    
    # 4. Connector Line
    last_row = history_df.iloc[-1]
    plt.plot([last_row['DATE_DT'], forecast_dt], [last_row['QUANTITY'], predicted_qty], 
             'r--', alpha=0.4)
    
    plt.title(f'WMS Forecast API - Brand: {brand} | Hier: {hierarchy}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    
    # Save artifact
    output_path = f"runs/api_forecast_{brand}_{hierarchy}.png"
    os.makedirs("runs", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path

# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Main prediction endpoint. 
    Constructs features on-the-fly and returns the forecast.
    """
    try:
        # 1. Context Retrieval
        history = get_historical_context(request.brand, request.hierarchy, request.date)
        
        # 2. Feature Engineering Pipeline
        current_day = pd.DataFrame([{
            'DATE': request.date,
            'BRAND': request.brand,
            'PRODUCTHIERARCHY3': request.hierarchy,
            'QUANTITY': 0  # Placeholder, será ignorado no lag
        }])
        
        full_context = pd.concat([history, current_day]).reset_index(drop=True)
        full_context['DATE'] = pd.to_datetime(full_context['DATE'])
        full_context = full_context.set_index('DATE')

        df_feat = add_lags(full_context)
        df_feat = add_diff(df_feat)
        df_feat = add_ewma(df_feat)
        df_feat = add_holidays(df_feat)
        df_feat = add_weekday_weekend_flags(df_feat)
        df_feat = add_calendar_events(df_feat)
        df_feat = create_features(df_feat)

        target_row = df_feat.tail(1).copy()
        
        cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
        for col in ['PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']:
            if col not in target_row.columns:
                target_row[col] = "unknown"
        
        X_transformed = preprocessor.transform(target_row)
        pred_log = model.predict(X_transformed)
        prediction = np.expm1(pred_log)[0]
        
        # 4. Artifact Generation
        plot_path = save_prediction_plot(
            request.brand, request.hierarchy, request.date, 
            round(float(prediction), 2), history
        )

        return {
            "brand": request.brand,
            "hierarchy": request.hierarchy,
            "forecast_date": request.date,
            "predicted_quantity": round(float(prediction), 2),
            "plot_path": plot_path
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))