"""
LSTM Training Module
------------------------------------------------
Trains a Long Short-Term Memory (LSTM) network for MULTI-STEP forecasting.

Key Changes:
- Target (y) is now a vector of 'forecast_horizon' days (e.g., 7 days).
- Uses Log Transformation on targets to handle sales spikes.
- Model output layer size equals 'forecast_horizon'.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import tensorflow.keras.backend as K

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def quantile_loss(q, y_true, y_pred):
    """
    Pinball Loss for Quantile Regression.
    q: The quantile to be estimated (e.g., 0.9 for 90th percentile).
    """
    e = y_true - y_pred
    return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

def create_sequences_multistep(X, y_data, time_steps=30, horizon=7):
    """
    Transforms 2D data into 3D sequences for Multi-Step LSTM.
    
    Structure:
    - Input (X): Window of 'time_steps' (e.g., 30 days).
    - Target (y): Vector of 'horizon' future steps (e.g., next 7 days).
    
    Args:
        X (np.array): Feature matrix (Scaled).
        y_data (np.array): Target vector (Scaled Log Quantity).
        time_steps (int): Lookback window size.
        horizon (int): Number of days to predict.
        
    Returns:
        tuple: (Xs, ys)
    """
    Xs, ys = [], []
    
    # We iterate until we run out of data for the horizon
    # Limit = Length - Lookback - Horizon + 1
    num_samples = len(X) - time_steps - horizon + 1
    
    for i in range(num_samples):
        # 1. Input: Days [i] to [i + 30]
        v_x = X[i:(i + time_steps)]
        Xs.append(v_x)
        
        # 2. Target: Days [i + 30] to [i + 30 + 7]
        # We grab the 'horizon' steps immediately following the input window
        v_y = y_data[i + time_steps : i + time_steps + horizon]
        ys.append(v_y.flatten()) # Flatten to ensure shape (7,)
        
    return np.array(Xs), np.array(ys)

def calc_metrics_real_scale(y_true_scaled, y_pred_scaled, scaler):
    """
    Inverse transforms scaled log data to real quantity and calculates metrics.
    Returns: Dict of metrics and Real Scale arrays for plotting.
    """
    # 1. Inverse Scale (0-1 -> Log Scale)
    y_true_log = scaler.inverse_transform(y_true_scaled)
    y_pred_log = scaler.inverse_transform(y_pred_scaled)
    
    # 2. Inverse Log (Log -> Real Quantity)
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.expm1(y_pred_log)
    
    # 3. Compute Global Metrics
    mae = np.mean(np.abs(y_true_real - y_pred_real))
    rmse = np.sqrt(np.mean((y_true_real - y_pred_real)**2))
    
    # 4. Compute Step-Specific Metrics
    # Day 1 MAE (First column)
    mae_day1 = np.mean(np.abs(y_true_real[:, 0] - y_pred_real[:, 0]))
    # Day 7 MAE (Last column)
    mae_day7 = np.mean(np.abs(y_true_real[:, -1] - y_pred_real[:, -1]))
    
    metrics = {
        "mae_global": mae,
        "rmse_global": rmse,
        "mae_day1": mae_day1,
        "mae_day7": mae_day7
    }
    
    return metrics

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_lstm(time_steps=30, forecast_horizon=7, epochs=20, batch_size=32, neurons=64, learning_rate=0.001, dropout=0.2, resume=False):
    """
    Trains the LSTM model using explicit 3D Numpy arrays.
    
    Args:
        time_steps (int): Number of past days to look back.
        forecast_horizon (int): Number of future days to predict.
        epochs (int): Number of training epochs.
        batch_size (int): Size of training batches.
        neurons (int): Number of LSTM neurons.
        learning_rate (float): Learning rate for Adam optimizer.
        dropout (float): Dropout rate for regularization.
        resume (bool): Whether to resume training from last checkpoint.
    """
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models/lstm')
    os.makedirs(models_dir, exist_ok=True)
    
    final_model_path = os.path.join(models_dir, 'lstm_final_model.keras')
    best_model_path = os.path.join(models_dir, 'lstm_best_checkpoint_model.keras')
    
    print(f"Starting LSTM Training")
    print(f"    Config: Lookback={time_steps}, Horizon={forecast_horizon}, Neurons={neurons}, Dropout={dropout}, LR={learning_rate}")
    
    # 1. Load Data
    # --------------------------------------------------------------------------
    train_df, val_df, _ = prepare_datasets()

    if train_df.empty:
        print("Error: No training data available.")
        return None

    # 2. Define Features
    # --------------------------------------------------------------------------
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
    num_cols = [f for f in features if f not in cat_cols]
    
    target_source_col = 'QUANTITY'

    # 3. Preprocessing (Scaling)
    # --------------------------------------------------------------------------
    # Neural Networks require input scaling (0-1).
    
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            # sparse_output=False because LSTMs generally need dense arrays
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ],
        verbose_feature_names_out=False
    )

    print("Fitting Preprocessor & Scaling Data...")
    
    # Process Features (X)
    X_train_processed = preprocessor.fit_transform(train_df[features])
    X_val_processed = preprocessor.transform(val_df[features])
    
    print("Applying Log Transformation to Target...")
    
    y_train_log = np.log1p(train_df[[target_source_col]])
    y_val_log = np.log1p(val_df[[target_source_col]])
    
    # Process Target (y) - Important for MSE loss stability
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(y_train_log)
    y_val_scaled = target_scaler.transform(y_val_log)

    # 4. Create Multi-Step Sequences
    # --------------------------------------------------------------------------
    print(f"Creating Multi-Step Sequences (X={time_steps}, y={forecast_horizon})...")
    
    X_train_seq, y_train_seq = create_sequences_multistep(X_train_processed, y_train_scaled, time_steps, forecast_horizon)
    X_val_seq, y_val_seq = create_sequences_multistep(X_val_processed, y_val_scaled, time_steps, forecast_horizon)

    print(f"    Train Input Shape: {X_train_seq.shape} (Samples, Steps, Features)")
    print(f"    Train Target Shape: {y_train_seq.shape} (Samples, Horizon)")

    # 5. Build or load Model Architecture
    # --------------------------------------------------------------------------
    model = None
    
    if resume:
        load_path = None
        if os.path.exists(final_model_path):
            load_path = final_model_path
        elif os.path.exists(best_model_path):
            load_path = best_model_path
            
        if load_path:
            print(f"RESUME: Loading existing model from {load_path}...")
            try:
                model = tf.keras.models.load_model(load_path, compile=False)
                print("RESUME: Model loaded successfully.")
            except Exception as e:
                print(f"RESUME: Failed to load model ({e}). Building new one.")
        else:
            print("RESUME: No existing model found. Building new one.")
    if model is None:
        model = tf.keras.models.Sequential([
            # Input Layer
            tf.keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            
            # LSTM Layer
            tf.keras.layers.LSTM(neurons, return_sequences=False, activation='tanh'),
            
            # Dropout for regularization
            tf.keras.layers.Dropout(dropout),
            
            # Output Layer
            tf.keras.layers.Dense(forecast_horizon)
        ])
    
    quantile = 0.80
    
    print("Starting MLflow Run...")
    with mlflow.start_run(run_name="LSTM_Multistep"):
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                    loss=lambda y, f: quantile_loss(quantile, y, f), 
                    metrics=['mae'])
        
        mlflow.log_params({
            "model_type": "lstm_multistep",
            "time_steps": time_steps,
            "forecast_horizon": forecast_horizon,
            "neurons": neurons,
            "epochs": epochs,
            "batch_size": batch_size,
            "dropout": dropout,
            "learning_rate": learning_rate
        })

        # 6. Train
        # --------------------------------------------------------------------------
        print("Fitting LSTM model...")
        
        # Directory for checkpoints
        checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'lstm', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Checkpoint: Save best model based on Val Loss
        checkpoint_path = os.path.join(models_dir, 'lstm_best_checkpoint_model.keras')
        
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,     # Only save when model improves
            monitor='val_loss',      # Monitor validation loss
            mode='min',              # Lower is better
            verbose=1
        )
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[early_stop, checkpoint_cb],
            verbose=1
        )
        
        print("Calculating Real Scale Metrics...")
        y_val_pred_scaled = model.predict(X_val_seq, verbose=0)
        
        metrics = calc_metrics_real_scale(y_val_seq, y_val_pred_scaled, target_scaler)
        
        print(f"Metrics: MAE={metrics['mae_global']:.2f}, Day1={metrics['mae_day1']:.2f}, Day7={metrics['mae_day7']:.2f}")
        mlflow.log_metrics(metrics)
        
        print("Logging Model and Artifacts...")

        # 7. Save Artifacts
        # --------------------------------------------------------------------------
        local_model_path = os.path.join(models_dir, 'lstm_final_model.keras')
        model.save(local_model_path)

        joblib.dump(preprocessor, os.path.join(models_dir, 'lstm_preprocessor.joblib'))
        joblib.dump(target_scaler, os.path.join(models_dir, 'lstm_target_scaler.joblib'))
        
        metadata = {
            'time_steps': time_steps, 
            'forecast_horizon': forecast_horizon
        }
        joblib.dump(metadata, os.path.join(models_dir, 'lstm_metadata.joblib'))
        
        mlflow.log_artifact(local_model_path)
        mlflow.log_artifact(os.path.join(models_dir, 'lstm_preprocessor.joblib'))
        mlflow.log_artifact(os.path.join(models_dir, 'lstm_target_scaler.joblib'))
        mlflow.log_artifact(os.path.join(models_dir, 'lstm_metadata.joblib'))

        print(f"Success! Run ID: {mlflow.active_run().info.run_id}")
    
    print("Model, Preprocessor, Target Scaler and Metadata saved.")
    return model

if __name__ == "__main__":
    train_lstm()