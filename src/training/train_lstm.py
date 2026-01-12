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

# Import Sklearn tools
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
    Pinball Loss para Quantile Regression.
    q: O quantil desejado (ex: 0.90 para ser agressivo/otimista).
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

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_lstm(time_steps=30, forecast_horizon=7, epochs=20, batch_size=32, neurons=64, learning_rate=0.001, dropout=0.2):
    """
    Trains the LSTM model using explicit 3D Numpy arrays.
    
    Args:
        use_filtering (bool): If True, uses only Top Brands/Hierarchies data.
    """
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

    # 5. Build Model Architecture
    # --------------------------------------------------------------------------
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

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss=lambda y, f: quantile_loss(quantile, y, f), 
                  metrics=['mae'])

    # 6. Train
    # --------------------------------------------------------------------------
    print("Fitting LSTM model...")
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stop],
        verbose=1
    )

    # 7. Save Artifacts
    # --------------------------------------------------------------------------
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    model.save(os.path.join(models_dir, 'lstm_model.keras'))
    
    joblib.dump(preprocessor, os.path.join(models_dir, 'lstm_preprocessor.joblib'))
    joblib.dump(target_scaler, os.path.join(models_dir, 'lstm_target_scaler.joblib'))
    
    metadata = {
        'time_steps': time_steps, 
        'forecast_horizon': forecast_horizon
    }
    joblib.dump(metadata, os.path.join(models_dir, 'lstm_metadata.joblib'))
    
    print("Model, Preprocessor, Target Scaler and Metadata saved.")
    return model

if __name__ == "__main__":
    train_lstm()