"""
LSTM Evaluation Module
-------------------------------------------
Evaluates the trained LSTM model on test data for Multi-Step forecasting.
1. Loads model, scalers, and metadata (horizon).
2. Creates (N, time_steps) inputs and (N, forecast_horizon) targets.
3. Predicts and Inverse Log-Transforms back to real units.
4. Computes metrics (MAE, RMSE) overall AND per forecast day (t+1 vs t+forecast_horizon).
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

def create_sequences_for_eval(df, X_scaled, y_scaled, time_steps=30, horizon=7):
    """
    Creates sequences for evaluation.
    Target y is now a matrix (N, horizon).
    Metadata corresponds to the 'Forecast Date' (the last day of input window).
    """
    Xs, ys = [], []
    metadata = [] 
    
    # Need enough data for Input + Horizon
    if len(X_scaled) < time_steps + horizon:
        return np.array(Xs), np.array(ys), pd.DataFrame(metadata)

    # Note: We iterate so that we have 'horizon' targets available
    num_samples = len(X_scaled) - time_steps - horizon + 1

    for i in range(num_samples):
        # 1. Input: Window of 'time_steps'
        v_x = X_scaled[i:(i + time_steps)]
        Xs.append(v_x)
        
        # 2. Target: The next 'horizon' days
        v_y = y_scaled[i + time_steps : i + time_steps + horizon]
        ys.append(v_y.flatten()) # Shape becomes (horizon,)
        
        # 3. Metadata (Associated with the day we make the prediction)
        # The "Forecast Date" is the last day of the input window (i + time_steps - 1)
        row_idx = i + time_steps - 1
        row = df.iloc[row_idx]
        
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

def evaluate_lstm(test_data, time_steps=30, forecast_horizon=7):
    """
    Runs the full evaluation pipeline for Multi-Step LSTM.
    """
    # 1. Load Artifacts
    MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models/lstm')
    MODEL_PATH = os.path.join(MODEL_DIR, 'lstm_model.keras')
    PREPROC_PATH = os.path.join(MODEL_DIR, 'lstm_preprocessor.joblib')
    TARGET_SCALER_PATH = os.path.join(MODEL_DIR, 'lstm_target_scaler.joblib')
    META_PATH = os.path.join(MODEL_DIR, 'lstm_metadata.joblib')

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return None

    # Load Metadata if available to overwrite args
    if os.path.exists(META_PATH):
        try:
            meta = joblib.load(META_PATH)
            time_steps = meta.get('time_steps', time_steps)
            forecast_horizon = meta.get('forecast_horizon', forecast_horizon)
            print(f"Loaded Metadata: Lookback={time_steps}, Horizon={forecast_horizon}")
        except:
            print("Warning: Could not load metadata. Using arguments.")

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    print("Loading scalers...")
    preprocessor = joblib.load(PREPROC_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2',
                'is_black_friday_week', 
                'is_pre_christmas', 
                'is_post_holiday_slump', 
                'is_payday_zone',
                'days_to_christmas']
    
    target_source_col = 'QUANTITY'
    
    # 2. Preprocess Test Data
    print("Preprocessing Test Data...")
    
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    for c in cat_cols:
        if c in test_data.columns:
            test_data[c] = test_data[c].astype(str)

    # Transform Features (X)
    X_test_processed = preprocessor.transform(test_data[features]).astype(np.float32)
    
    # Transform Target (y) - LOG -> SCALE
    # Use RAW Quantity column, then Log, then Scale
    y_raw_col = test_data[[target_source_col]].values
    y_log = np.log1p(y_raw_col)
    y_scaled = target_scaler.transform(y_log).astype(np.float32)

    # 3. Create Sequences
    print(f"Creating Sequences (Horizon: {forecast_horizon})...")
    X_seq, y_seq, meta_df = create_sequences_for_eval(test_data, X_test_processed, y_scaled, time_steps, forecast_horizon)
    
    if len(X_seq) == 0:
        print("Error: Test data is too short for the configured window + horizon.")
        return None

    # 4. Predict
    print("Generating Predictions...")
    y_pred_scaled = model.predict(X_seq, verbose=1)

    # 5. Inverse Transform (Back to Real Scale)
    print("Inverse Scaling & Log Transformation...")
    # Inverse Scale
    y_pred_log = target_scaler.inverse_transform(y_pred_scaled)
    y_true_log = target_scaler.inverse_transform(y_seq)
    
    # Inverse Log (Expm1)
    y_pred_raw = np.expm1(y_pred_log)
    y_true_raw = np.expm1(y_true_log)

    # 6. Compute Metrics
    print("Calculating Metrics (Averaged over Horizon)...")
    
    rows = []
    
    # Helper to calculate metrics for a pair of matrices
    def calc_metrics(y_t, y_p):
        mae = np.mean(np.abs(y_p - y_t))
        rmse = np.sqrt(np.mean((y_p - y_t)**2))
        return mae, rmse, np.mean(y_t), np.mean(y_p)

    for (bcode, hcode), idxs in meta_df.groupby(['BRAND', 'PRODUCTHIERARCHY3']).groups.items():
        # Filter arrays by the indices belonging to this group
        y_t_group = y_true_raw[idxs]
        y_p_group = y_pred_raw[idxs]
        
        if len(y_t_group) == 0: continue

        # 1. Overall Metrics (Flattened)
        mae, rmse, mean_act, mean_pred = calc_metrics(y_t_group, y_p_group)
        
        # 2. Per-Step Metrics (Day 1 vs Day 7)
        # y_t_group is shape (Samples, Horizon)
        mae_day1 = np.mean(np.abs(y_p_group[:, 0] - y_t_group[:, 0]))
        mae_day7 = np.mean(np.abs(y_p_group[:, -1] - y_t_group[:, -1]))
        
        # Bias (Total Volume Error)
        bias = np.mean(y_p_group - y_t_group)

        # SMAPE
        smape_val = _smape(y_t_group, y_p_group) * 100
        
        rows.append({
            "BRAND": bcode,
            "PRODUCTHIERARCHY3": hcode,
            "n_points": len(y_t_group), # Number of sequences, not total days
            "mae_avg": mae,
            "mae_day1": mae_day1,
            "mae_day7": mae_day7,
            "rmse": rmse,
            "smape_%": smape_val,
            "mean_actual": mean_act,
            "mean_pred": mean_pred,
            "bias_avg": bias
        })

    metrics_df = pd.DataFrame(rows)

    # Global Metrics
    if len(y_true_raw) > 0:
        mae, rmse, mean_act, mean_pred = calc_metrics(y_true_raw, y_pred_raw)
        overall = {
            "BRAND": "__ALL__",
            "PRODUCTHIERARCHY3": "__ALL__",
            "n_points": len(y_true_raw),
            "mae_avg": mae,
            "mae_day1": np.mean(np.abs(y_pred_raw[:, 0] - y_true_raw[:, 0])),
            "mae_day7": np.mean(np.abs(y_pred_raw[:, -1] - y_true_raw[:, -1])),
            "rmse": rmse,
            "smape_%": _smape(y_true_raw, y_pred_raw) * 100,
            "mean_actual": mean_act,
            "mean_pred": mean_pred,
            "bias_avg": np.mean(y_pred_raw - y_true_raw)
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([overall])], ignore_index=True)

    return metrics_df.sort_values(['BRAND', 'PRODUCTHIERARCHY3'])

if __name__ == "__main__":
    from src.utils.data_split import prepare_datasets
    print("Loading Data...")
    _, _, test_data = prepare_datasets()
    metrics = evaluate_lstm(test_data)
    print(metrics.tail())