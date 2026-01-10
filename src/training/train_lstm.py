"""
LSTM Training Module (Standard Implementation)
----------------------------------------------
Trains a Long Short-Term Memory (LSTM) neural network for 1-day forecasting.

Approach:
- Uses a standard sliding window technique to create 3D arrays (N, T, F).
- Loads data into memory (RAM) for faster training loop execution.
- Predicts the target value associated with the last step of the input window.
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

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_sequences(X, y, time_steps=30):
    """
    Transforms 2D data into 3D sequences for LSTM consumption.
    
    Structure:
    - Input (X): A sliding window of historical features of length 'time_steps'.
    - Target (y): The target value corresponding to the LAST step of that window.
      (Since our dataframe already has 'quantity_next_day' aligned, we take the 
       target from the last row of the sequence).
    
    Args:
        X (np.array): Feature matrix (2D).
        y (np.array): Target vector (1D).
        time_steps (int): Lookback window size.
        
    Returns:
        tuple: (Xs, ys).
    """
    Xs, ys = [], []
    
    # Iterate through the array to create sequences
    # We stop when i + time_steps exceeds the array length
    for i in range(len(X) - time_steps + 1):
        # Slice the window [i : i+30]
        v = X[i:(i + time_steps)]
        Xs.append(v)
        
        # Taking the target from the last step of the window
        # Because row 't' in our DF already contains the target 'next_day' for t.
        ys.append(y[i + time_steps - 1])
        
    return np.array(Xs), np.array(ys)

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_lstm(time_steps=30, epochs=20, batch_size=32, neurons=64, learning_rate=0.001, dropout=0.2):
    """
    Trains the LSTM model using explicit 3D Numpy arrays.
    
    Args:
        use_filtering (bool): If True, uses only Top Brands/Hierarchies data.
    """
    print(f"Starting LSTM Training (Standard)")
    print(f"    Config: Lookback={time_steps}, Neurons={neurons}, Dropout={dropout}, LR={learning_rate}")
    
    # 1. Load Data
    # --------------------------------------------------------------------------
    train_df, val_df, test_df = prepare_datasets()

    if train_df.empty:
        print("Error: No training data available.")
        return None

    # 2. Define Features
    # --------------------------------------------------------------------------
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    num_cols = [f for f in features if f not in cat_cols]
    
    target_col = 'quantity_next_day'

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
    
    y_train_log = np.log1p(train_df[[target_col]])
    y_val_log = np.log1p(val_df[[target_col]])
    
    # Process Target (y) - Important for MSE loss stability
    target_scaler = MinMaxScaler()
    y_train_processed = target_scaler.fit_transform(y_train_log)
    y_val_processed = target_scaler.transform(y_val_log)

    # 4. Create 3D Sequences
    # --------------------------------------------------------------------------
    print(f"Creating Sequences (Time Steps: {time_steps})...")
    
    X_train_seq, y_train_seq = create_sequences(X_train_processed, y_train_processed, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_processed, y_val_processed, time_steps)

    print(f"    Train Input Shape: {X_train_seq.shape} (Samples, Steps, Features)")
    print(f"    Val Input Shape:   {X_val_seq.shape}")

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
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
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
    
    print("Model, Preprocessor, and Target Scaler saved.")
    return model

if __name__ == "__main__":
    train_lstm()