"""
WMS Forecast API Module
-----------------------
FastAPI implementation for real-time demand forecasting using XGBoost and LSTM.
Standardized visualization for both models.
"""
import pandas as pd
import numpy as np
import sqlite3
import joblib
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime, timedelta

# Import project feature utilities
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
# ARTIFACT LOADING & PATHS
# ==============================================================================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DB_PATH = os.path.join(ROOT_DIR, 'data/api_forecast.db')

# XGBoost Paths
XGB_MODEL_PATH = os.path.join(ROOT_DIR, 'models/xgboost/xgboost_final_model.joblib')
XGB_PREPROC_PATH = os.path.join(ROOT_DIR, 'models/xgboost/preprocessor.joblib')

# LSTM Paths
LSTM_MODEL_PATH = os.path.join(ROOT_DIR, 'models/lstm/lstm_model.keras')
LSTM_PREPROC_PATH = os.path.join(ROOT_DIR, 'models/lstm/lstm_preprocessor.joblib')
LSTM_SCALER_PATH = os.path.join(ROOT_DIR, 'models/lstm/lstm_target_scaler.joblib')

# Load Artifacts
xgb_model = joblib.load(XGB_MODEL_PATH) if os.path.exists(XGB_MODEL_PATH) else None
xgb_preprocessor = joblib.load(XGB_PREPROC_PATH) if os.path.exists(XGB_PREPROC_PATH) else None

lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False) if os.path.exists(LSTM_MODEL_PATH) else None
lstm_preprocessor = joblib.load(LSTM_PREPROC_PATH) if os.path.exists(LSTM_PREPROC_PATH) else None
lstm_target_scaler = joblib.load(LSTM_SCALER_PATH) if os.path.exists(LSTM_SCALER_PATH) else None

# ==============================================================================
# SCHEMA & DATA UTILITIES
# ==============================================================================
class PredictRequest(BaseModel):
    brand: str
    hierarchy: str
    date: str

def get_historical_context(brand, hier, date, limit=60):
    """Retrieves historical data from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT DATE, BRAND, PRODUCTHIERARCHY3, QUANTITY 
        FROM history 
        WHERE BRAND = ? AND PRODUCTHIERARCHY3 = ? AND DATE < ?
        ORDER BY DATE DESC LIMIT ?
    """
    df = pd.read_sql(query, conn, params=(brand, hier, date, limit))
    conn.close()
    return df.sort_values('DATE')

# ==============================================================================
# UNIFIED VISUALIZATION LOGIC
# ==============================================================================
def save_standardized_plot(brand, hierarchy, forecast_dates, forecast_values, history_df, model_name):
    """Generates a plot for both models (XGBoost/LSTM)."""
    plt.figure(figsize=(10, 5))
    
    # 1. Process History
    hist_df = history_df.copy()
    hist_df['DATE_DT'] = pd.to_datetime(hist_df['DATE'])
    hist_df = hist_df.sort_values('DATE_DT').tail(30)
    
    # Plot History
    plt.plot(hist_df['DATE_DT'], hist_df['QUANTITY'], 
             label='History', marker='o', color='#1f77b4', alpha=0.6)
    
    # 2. Process Forecast
    forecast_dates_dt = pd.to_datetime(forecast_dates)
    plt.plot(forecast_dates_dt, forecast_values, 
             label=f'{model_name} Forecast', color='red', marker='o', 
             linestyle='--' if len(forecast_values) > 1 else '', alpha=0.8, zorder=5)
    
    # 3. Visual Connector
    last_hist = hist_df.iloc[-1]
    plt.plot([last_hist['DATE_DT'], forecast_dates_dt[0]], [last_hist['QUANTITY'], forecast_values[0]], 
             color='red', linestyle='--', alpha=0.4)
    
    # Formatting
    plt.title(f'WMS Forecast API ({model_name}) - Brand: {brand} | Hier: {hierarchy}')
    plt.xlabel('Date')
    plt.ylabel('Quantity')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xticks(rotation=35)
    plt.legend()
    plt.tight_layout()
    
    # Save Artifact
    output_path = f"runs/api_forecast_{model_name.lower()}_{brand}_{hierarchy}.png"
    os.makedirs("runs", exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    return output_path

# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.post("/predict/xgboost")
async def predict_xgboost(request: PredictRequest):
    """XGBoost next-day prediction."""
    try:
        if not xgb_model: raise RuntimeError("XGBoost artifacts not found.")
        
        history = get_historical_context(request.brand, request.hierarchy, request.date)
        current_day = pd.DataFrame([{'DATE': request.date, 'BRAND': request.brand, 
                                     'PRODUCTHIERARCHY3': request.hierarchy, 'QUANTITY': 0}])
        
        # Feature Engineering
        full_context = pd.concat([history, current_day]).reset_index(drop=True)
        full_context['DATE'] = pd.to_datetime(full_context['DATE'])
        df_feat = add_lags(full_context.set_index('DATE'))
        df_feat = add_diff(df_feat); df_feat = add_ewma(df_feat)
        df_feat = add_holidays(df_feat); df_feat = add_weekday_weekend_flags(df_feat)
        df_feat = add_calendar_events(df_feat); df_feat = create_features(df_feat)

        target_row = df_feat.tail(1).copy()
        for col in ['PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']:
            if col not in target_row.columns: target_row[col] = "unknown"
        
        # Inference
        X_transformed = xgb_preprocessor.transform(target_row)
        prediction = np.expm1(xgb_model.predict(X_transformed))[0]
        
        plot_path = save_standardized_plot(request.brand, request.hierarchy, [request.date], 
                                          [round(float(prediction), 2)], history, "XGBoost")

        return {"brand": request.brand, "hierarchy": request.hierarchy, "predicted_quantity": round(float(prediction), 2), "plot_path": plot_path}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/lstm")
async def predict_lstm(request: PredictRequest):
    """LSTM Multi-step prediction (e.g., 7 days)."""
    try:
        if not lstm_model: raise RuntimeError("LSTM artifacts not found.")
        
        history = get_historical_context(request.brand, request.hierarchy, request.date, limit=60)
        if len(history) < 30: raise HTTPException(status_code=400, detail="Insufficient history for LSTM (min 30 days)")

        # Feature Engineering for 30-day window
        df_feat = history.copy()
        df_feat['DATE'] = pd.to_datetime(df_feat['DATE'])
        df_feat = df_feat.set_index('DATE')
        df_feat = add_lags(df_feat); df_feat = add_diff(df_feat); df_feat = add_ewma(df_feat)
        df_feat = add_holidays(df_feat); df_feat = add_weekday_weekend_flags(df_feat)
        df_feat = add_calendar_events(df_feat); df_feat = create_features(df_feat)

        features_list = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                         'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                         'is_weekend', 'is_portuguese_holiday', 'BRAND', 'PRODUCTHIERARCHY3', 
                         'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2', 'is_black_friday_week', 
                         'is_pre_christmas', 'is_post_holiday_slump', 'is_payday_zone', 'days_to_christmas']

        for col in ['PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']:
            if col not in df_feat.columns: df_feat[col] = "unknown"

        # Prepare 3D Input
        X_scaled = lstm_preprocessor.transform(df_feat[features_list].tail(30)).astype(np.float32)
        y_pred_scaled = lstm_model.predict(np.expand_dims(X_scaled, axis=0))
        y_pred_real = np.expm1(lstm_target_scaler.inverse_transform(y_pred_scaled))[0]

        # Generate Forecast Dates
        forecast_dates = [(pd.to_datetime(request.date) + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(len(y_pred_real))]
        plot_path = save_standardized_plot(request.brand, request.hierarchy, forecast_dates, y_pred_real, history, "LSTM")

        return {"brand": request.brand, "hierarchy": request.hierarchy, "forecast_horizon": [{"date": d, "qty": round(float(v), 2)} for d, v in zip(forecast_dates, y_pred_real)], "plot_path": plot_path}
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))