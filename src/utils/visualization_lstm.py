"""
LSTM Visualization Utility
-----------------------------------------------
Visualizes forecasts for a specific Product Hierarchy and Brand.
Handles Multi-Step (e.g., 7-day horizon) output by plotting distinct forecast trajectories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# CONFIGURATION
# ==============================================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR = os.path.join(ROOT_DIR, 'models/lstm')

# Paths
MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
PREPROC_PATH = os.path.join(MODEL_DIR, 'lstm_preprocessor.joblib')
TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_target_scaler.joblib')
META_PATH = os.path.join(MODEL_DIR, 'lstm_metadata.joblib')

# ==============================================================================
# HELPER: SEQUENCE CREATION
# ==============================================================================

def create_plot_sequences(X_scaled, time_steps=30):
    """
    Creates input sequences for plotting.
    """
    Xs = []
    for i in range(len(X_scaled) - time_steps + 1):
        v = X_scaled[i:(i + time_steps)]
        Xs.append(v)
    return np.array(Xs)

# ==============================================================================
# MAIN VISUALIZATION LOGIC
# ==============================================================================

def visualize_forecast_lstm(hier_code, brand_code):
    """
    Generates Multi-Step LSTM forecast plot for a specific product-brand.
    """
    
    # 1. Load Artifacts & Metadata
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading LSTM model, scalers, and metadata...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    preprocessor = joblib.load(PREPROC_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    time_steps = 30
    forecast_horizon = 7

    if os.path.exists(META_PATH):
        try:
            meta = joblib.load(META_PATH)
            time_steps = meta.get('time_steps', time_steps)
            forecast_horizon = meta.get('forecast_horizon', forecast_horizon)
            print(f"    Metadata loaded: Lookback={time_steps}, Horizon={forecast_horizon}")
        except:
            print("Warning: Could not load metadata. Using defaults.")
    else:
        print("Warning: Metadata file not found. Using defaults.")

    # 2. Load Data
    print("Loading test data...")
    _, _, test_data = prepare_datasets()

    mask = (test_data['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)) & \
           (test_data['BRAND'].astype(str) == str(brand_code))

    df_viz = test_data[mask].copy()
    
    if df_viz.empty:
        print(f"Warning: No data found for H: {hier_code} / B: {brand_code}")
        return

    if df_viz.index.name == 'DATE':
        df_viz.index = pd.to_datetime(df_viz.index)
        df_viz = df_viz.sort_index()
    elif 'DATE' in df_viz.columns:
        df_viz['DATE'] = pd.to_datetime(df_viz['DATE'])
        df_viz = df_viz.set_index('DATE')
        df_viz = df_viz.sort_index()
    else:
        df_viz.index = pd.to_datetime(df_viz.index)
        df_viz = df_viz.sort_index()
    # -----------------------------------------

    if len(df_viz) < time_steps + forecast_horizon:
        print(f"Error: Not enough data ({len(df_viz)}) for lookback + horizon.")
        return

    # 3. Preprocessing
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2',
                'is_black_friday_week', 
                'is_pre_christmas', 
                'is_post_holiday_slump', 
                'is_payday_zone',
                'days_to_christmas']
    
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    for c in cat_cols:
        df_viz[c] = df_viz[c].astype(str)

    X_viz_scaled = preprocessor.transform(df_viz[features]).astype(np.float32)
    
    # 4. Generate Sequences & Predict
    print("Generating sequences and predictions...")
    X_seq = create_plot_sequences(X_viz_scaled, time_steps)
    
    num_valid_preds = len(df_viz) - time_steps - forecast_horizon + 1
    if num_valid_preds <= 0:
         print("Not enough data to plot.")
         return

    X_seq_valid = X_seq[:num_valid_preds]

    # A. Predict (Log Scale)
    y_pred_scaled = model.predict(X_seq_valid, verbose=0)
    
    N, H = y_pred_scaled.shape
    y_pred_log = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(N, H)
    
    # B. Calculate Sigma (Correction Factor)
    true_vals = []
    
    for i in range(len(y_pred_log)):
        idx = i + time_steps
        if idx < len(df_viz):
            true_vals.append(df_viz['QUANTITY'].iloc[idx])
            
    true_vals = np.array(true_vals)
    
    true_vals_log = np.log1p(true_vals)
    preds_log_day1 = y_pred_log[:len(true_vals), 0]
    
    residuals = true_vals_log - preds_log_day1
    sigma2 = np.var(residuals)
    
    print(f"Log-Space Variance (Sigma^2): {sigma2:.4f}")
    
    # C. Apply Correction: exp(pred + sigma^2/2) - 1
    y_pred_raw = np.expm1(y_pred_log + (sigma2 / 2.0))
    # y_pred_raw = np.expm1(y_pred_log)

    # 5. Plotting (Spaghetti/Fan Plot)
    print("Generating plot...")
    
    desc = df_viz['PRODUCTHIERARCHY3NAME'].iloc[0] if 'PRODUCTHIERARCHY3NAME' in df_viz else "Unknown"
    brand_desc = df_viz['BRANDNAME'].iloc[0] if 'BRANDNAME' in df_viz else "Unknown"

    plt.figure(figsize=(14, 7))
    
    # Plot Actual
    plot_start_idx = max(0, time_steps - 60)
    plot_df = df_viz.iloc[plot_start_idx:]
    plt.plot(plot_df.index, plot_df['QUANTITY'], label='Actual Sales', color='black', alpha=0.4, linewidth=2)

    # Plot Forecast Trajectories
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
    num_trajectories_to_plot = 4
    
    pred_indices = range(len(y_pred_raw) - 1, -1, -forecast_horizon)[:num_trajectories_to_plot]
    
    for i, pred_idx in enumerate(pred_indices):
        forecast_vector = y_pred_raw[pred_idx]
        
        start_date_idx = pred_idx + time_steps
        end_date_idx = start_date_idx + forecast_horizon
        
        # Ensure we don't go out of bounds
        if end_date_idx > len(df_viz):
            continue
            
        forecast_dates = df_viz.index[start_date_idx : end_date_idx]
        
        color = colors[i % len(colors)]
        start_date_str = forecast_dates[0].strftime('%Y-%m-%d')
        plt.plot(forecast_dates, forecast_vector, 
                 label=f'{forecast_horizon}-Day Forecast (from {start_date_str})',
                 color=color, linewidth=2.5, marker='o', markersize=5, alpha=0.8, linestyle='--')

    plt.title(f'Multi-Step LSTM Forecast ({forecast_horizon} Days)\nHierarchy: {hier_code} ({desc}) | Brand: {brand_code} ({brand_desc})')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=7))
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()

    # 6. Save
    clean_brand = "".join(x for x in brand_desc if x.isalnum() or x in " -_").strip().replace(" ", "_")
    clean_hier = "".join(x for x in desc if x.isalnum() or x in " -_").strip().replace(" ", "_")
    
    filename = f'lstm_multistep_{forecast_horizon}d_{clean_brand}_{clean_hier}.png'
    output_path = os.path.join(ROOT_DIR, 'runs', filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Multi-step plot saved to: {output_path}")

if __name__ == "__main__":
    visualize_forecast_lstm('1060000100001.0', '1487.0')