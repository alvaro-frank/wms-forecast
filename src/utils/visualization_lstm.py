"""
LSTM Visualization Utility
--------------------------
Visualizes forecasts for a specific Product Hierarchy and Brand using the LSTM model.
1. Loads Keras model and Scalers.
2. Filters test data and creates sliding window sequences (3D).
3. Predicts and Inverse Scales back to real units.
4. Plots Actual vs Predicted.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

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
    Creates 3D sequences specifically for plotting.
    Returns the sequences AND the indices (to map back to dates).
    """
    Xs = []
    indices = []
    
    for i in range(len(X_scaled) - time_steps):
        v = X_scaled[i:(i + time_steps)]
        Xs.append(v)
        indices.append(i + time_steps - 1)
        
    return np.array(Xs), indices

# ==============================================================================
# MAIN VISUALIZATION LOGIC
# ==============================================================================

def visualize_forecast_lstm(hier_code, brand_code):
    """
    Generates LSTM forecast plot for a specific product-brand.
    """
    
    # 1. Load Artifacts
    # --------------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading LSTM model and scalers...")
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROC_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)
    
    # Try to load time_steps from metadata, default to 30
    time_steps = 30
    if os.path.exists(META_PATH):
        try:
            meta = joblib.load(META_PATH)
            time_steps = meta.get('time_steps', 30)
        except:
            pass
    print(f"    Using Time Steps (Lookback): {time_steps}")

    # 2. Load & Filter Data
    # --------------------------------------------------------------------------
    print("Loading test data...")
    # Assume filtering was used in training, so we load filtered test data
    _, _, test_data = prepare_datasets()

    mask = (test_data['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)) & \
           (test_data['BRAND'].astype(str) == str(brand_code))

    df_viz = test_data[mask].copy()
    
    if df_viz.empty:
        print(f"Warning: No data found for H: {hier_code} / B: {brand_code}")
        return
    
    # Sort and Set Index
    if 'DATE' in df_viz.columns:
        df_viz = df_viz.set_index('DATE')
    df_viz = df_viz.sort_index()

    # Check minimum length
    if len(df_viz) <= time_steps:
        print(f"Error: Not enough data points ({len(df_viz)}) for lookback window ({time_steps}).")
        return

    # 3. Preprocessing
    # --------------------------------------------------------------------------
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    for c in cat_cols:
        df_viz[c] = df_viz[c].astype(str)

    # Transform Features (X)
    X_viz_scaled = preprocessor.transform(df_viz[features]).astype(np.float32)
    
    # 4. Generate Sequences & Predict
    # --------------------------------------------------------------------------
    print("Generating sequences and predictions...")
    X_seq, valid_indices = create_plot_sequences(X_viz_scaled, time_steps)
    
    # Predict (Output is 0-1)
    y_pred_scaled = model.predict(X_seq, verbose=0)
    
    # Inverse Scale to Log Quantity
    y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    
    # Inverse Log Transform to Raw Quantity
    y_pred_raw = np.expm1(y_pred_log).flatten()

    # 5. Align Data for Plotting
    # --------------------------------------------------------------------------
    # We can only plot dates corresponding to valid_indices
    plot_dates = df_viz.index[valid_indices]
    y_actual_raw = df_viz['quantity_next_day'].iloc[valid_indices].values

    # Get names for title
    desc = df_viz['PRODUCTHIERARCHY3NAME'].iloc[0] if 'PRODUCTHIERARCHY3NAME' in df_viz else "Unknown"
    brand_desc = df_viz['BRANDNAME'].iloc[0] if 'BRANDNAME' in df_viz else "Unknown"

    # 6. Plot
    # --------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    
    plt.plot(plot_dates, y_actual_raw, label='Actual', alpha=0.7)
    plt.plot(plot_dates, y_pred_raw, label='LSTM Prediction', linestyle='--', color='darkorange')
    
    plt.title(f'LSTM Forecast (Lookback: {time_steps})\nHierarchy: {hier_code} ({desc})\nBrand: {brand_code} ({brand_desc})')
    plt.xlabel('Date')
    plt.ylabel('Quantity Next Day')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 7. Save
    # --------------------------------------------------------------------------
    clean_brand = "".join(x for x in brand_desc if x.isalnum() or x in " -_").strip().replace(" ", "_")
    clean_hier = "".join(x for x in desc if x.isalnum() or x in " -_").strip().replace(" ", "_")
    
    filename = f'lstm_forecast_{clean_brand}_{clean_hier}.png'
    output_path = os.path.join(ROOT_DIR, 'runs', filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    # Test
    visualize_forecast_lstm('1060000100001.0', '1487.0')