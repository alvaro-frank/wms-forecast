"""
LSTM Evaluation Module
----------------------
Evaluates the trained LSTM model on test data.
1. Loads the saved model and scalers.
2. Preprocesses data and creates 3D sequences (Time Steps).
3. Generates predictions and inverse-scales them back to real quantities.
4. Computes performance metrics (MAE, RMSE, MAPE) per Product-Brand pair.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _smape(y_true, y_pred, eps=1e-8):
    """Calculates Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def create_sequences_for_eval(df, X_scaled, y_scaled, time_steps=30):
    """
    Creates sequences for evaluation, keeping track of metadata (Brand, Hierarchy, Date).
    We need to map each 3D sequence back to its original identity to group metrics later.
    """
    Xs, ys = [], []
    metadata = [] # To store (Date, Brand, Hierarchy) for each prediction
    
    # We need to access the original dataframe to get metadata
    # The X_scaled array is parallel to df (assuming no shuffling)
    
    # Check alignment
    if len(df) != len(X_scaled):
        raise ValueError("Dataframe and Scaled Array length mismatch.")

    # Iterate
    # Note: We can only predict starting from index 'time_steps'
    for i in range(len(X_scaled) - time_steps + 1):
        # 1. Input Sequence
        v = X_scaled[i:(i + time_steps)]
        Xs.append(v)
        
        # 2. Target (Last step of window)
        # y_scaled[i + time_steps - 1] corresponds to the 'next_day' target 
        # associated with the LAST day of the input window.
        target_idx = i + time_steps - 1
        ys.append(y_scaled[target_idx])
        
        # 3. Metadata (for the target day)
        # We want to know which product/brand this prediction belongs to
        row = df.iloc[target_idx]
        metadata.append({
            'DATE': row.name if isinstance(df.index, pd.DatetimeIndex) else row['DATE'],
            'BRAND': row['BRAND'],
            'PRODUCTHIERARCHY3': row['PRODUCTHIERARCHY3'],
            'BRANDNAME': row.get('BRANDNAME', ''),
            'PRODUCTHIERARCHY3NAME': row.get('PRODUCTHIERARCHY3NAME', '')
        })
        
    return np.array(Xs), np.array(ys), pd.DataFrame(metadata)

# ==============================================================================
# MAIN EVALUATION LOGIC
# ==============================================================================

def evaluate_lstm(test_data, time_steps=30):
    """
    Runs the full evaluation pipeline for LSTM.
    """
    # 1. Load Artifacts
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
    PREPROC_PATH = os.path.join(MODEL_DIR, 'lstm_preprocessor.joblib')
    TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_target_scaler.joblib')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Loading scalers...")
    preprocessor = joblib.load(PREPROC_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    # 2. Define Features
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    
    target_col = 'quantity_next_day'
    
    # 3. Preprocess Test Data
    print("Preprocessing Test Data...")
    
    # Filter columns
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    for c in cat_cols:
        if c in test_data.columns:
            test_data[c] = test_data[c].astype(str)

    # Transform Features (X)
    X_test_processed = preprocessor.transform(test_data[features]).astype(np.float32)
    
    # Transform Target (y)
    y_test_log = np.log1p(test_data[[target_col]])
    y_test_scaled = target_scaler.transform(y_test_log).astype(np.float32)

    # 4. Create Sequences
    print(f"Creating Sequences (Lookback: {time_steps})...")
    X_seq, y_seq, meta_df = create_sequences_for_eval(test_data, X_test_processed, y_test_scaled, time_steps)
    
    if len(X_seq) == 0:
        print("Error: Test data is smaller than the lookback window.")
        return None

    # 5. Predict
    print("Generating Predictions...")
    y_pred_scaled = model.predict(X_seq, verbose=1)

    # 6. Inverse Transform
    print("Inverse Scaling & Log Transformation...")
    y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    y_pred_raw = np.expm1(y_pred_log).flatten()
    
    y_true_log = target_scaler.inverse_transform(y_seq)
    y_true_raw = np.expm1(y_true_log).flatten()

    # 7. Compile Results DataFrame
    results_df = meta_df.copy()
    results_df['y_true'] = y_true_raw.flatten()
    results_df['y_pred'] = y_pred_raw.flatten()
    
    # 8. Compute Metrics per Brand/Hierarchy
    print("Calculating Metrics...")
    rows = []
    global_y, global_yhat = [], []

    for (bcode, hcode), g in results_df.groupby(['BRAND', 'PRODUCTHIERARCHY3']):
        y = g['y_true'].values
        yhat = g['y_pred'].values
        n = len(y)
        
        if n == 0: continue

        mae = np.mean(np.abs(yhat - y))
        rmse = np.sqrt(np.mean((yhat - y)**2))
        
        nz = y != 0
        mape = np.mean(np.abs((yhat[nz] - y[nz]) / y[nz])) * 100 if np.any(nz) else np.nan
        smape_val = _smape(y, yhat) * 100
        
        rows.append({
            "BRAND": bcode,
            "PRODUCTHIERARCHY3": hcode,
            "n_points": n,
            "mae": mae,
            "rmse": rmse,
            "mape_%": mape,
            "smape_%": smape_val,
            "mean_actual": np.mean(y),
            "mean_pred": np.mean(yhat)
        })
        
        global_y.append(y)
        global_yhat.append(yhat)

    metrics_df = pd.DataFrame(rows)

    # Global Metrics
    if global_y:
        Y = np.concatenate(global_y)
        Yh = np.concatenate(global_yhat)
        
        overall = {
            "BRAND": "__ALL__",
            "PRODUCTHIERARCHY3": "__ALL__",
            "n_points": len(Y),
            "mae": np.mean(np.abs(Yh - Y)),
            "rmse": np.sqrt(np.mean((Yh - Y)**2)),
            "mape_%": np.nan, # Global MAPE is usually misleading
            "smape_%": _smape(Y, Yh) * 100,
            "mean_actual": np.mean(Y),
            "mean_pred": np.mean(Yh)
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([overall])], ignore_index=True)

    return metrics_df.sort_values(['BRAND', 'PRODUCTHIERARCHY3'])

if __name__ == "__main__":
    # Test harness
    from src.utils.data_split import prepare_datasets
    print("Loading Data...")
    _, _, test_data = prepare_datasets(filter_data=True)
    metrics = evaluate_lstm(test_data, time_steps=30)
    print(metrics.tail())