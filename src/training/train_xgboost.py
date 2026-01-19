"""
XGBoost Training Module
-----------------------
Trains an XGBoost regressor for time-series forecasting.
Handles data loading, feature preprocessing (OneHotEncoding, Imputation),
model training with early stopping, and artifact saving.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import sys
from mlflow.models import infer_signature

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# HELPER METRICS
# ==============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculates MAE, RMSE, Bias, and SMAPE (in original scale)."""
    
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 1. Reverse Log Transformation (if you trained on logs)
    # Assuming your model predicts log1p, we need to reverse it for real metrics
    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred)
    
    # 2. Calculate Metrics
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    bias = np.mean(y_pred_real - y_true_real)
    
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    numerator = np.abs(y_pred_real - y_true_real)
    denominator = (np.abs(y_true_real) + np.abs(y_pred_real)) / 2.0
    # Avoid division by zero
    smape = np.mean(np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)) * 100

    return {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "smape": smape
    }

# ==============================================================================
# MAIN TRAINING LOGIC
# ==============================================================================

def train_xgboost(learning_rate=0.01, max_depth=10, n_estimators=10000):
    """
    Trains the XGBoost model using the configured hyperparameters.
    
    Args:
        learning_rate (float): Step size shrinkage used in update to prevent overfitting.
        max_depth (int): Maximum depth of a tree.
        n_estimators (int): Number of boosting rounds.
        use_filtering (bool): Whether to filter for top brands only.

    Returns:
        xgb.XGBRegressor: The trained model object.
    """
    
    mlflow.set_tracking_uri("file:./mlruns")
    
    # 1. Load Data
    print("Preparing data...")
    # Uses the robust split logic from utils
    train_data, val_data, _ = prepare_datasets()
    
    if train_data.empty:
        print("No training data available.")
        return None

    # 2. Define Features
    # Numerical features (Lag, Diff, EWMA, Cyclic Time)
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2',
                'is_black_friday_week', 
                'is_pre_christmas', 
                'is_post_holiday_slump', 
                'is_payday_zone',
                'days_to_christmas']
    
    # Categorical features (ID columns to be One-Hot Encoded)
    CAT_COLS = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    NUM_COLS = [f for f in features if f not in CAT_COLS]

    # 3. Setup Preprocessing Pipeline
    # Numerical: Simple Mean Imputation
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    # Combined Preprocessor: OneHot for Cats + Imputer for Nums
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", numeric_pipe, NUM_COLS),
        ],
        sparse_threshold=1.0
    )

    # 4. Prepare X and y Matrices
    # Filter DataFrames to only include relevant columns
    X_train_df = train_data[CAT_COLS + NUM_COLS].copy()
    y_train = np.log1p(train_data['quantity_next_day'].values)

    X_val_df = val_data[CAT_COLS + NUM_COLS].copy()
    y_val = np.log1p(val_data['quantity_next_day'].values)

    # 5. Fit & Transform Data
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    
    checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'xgboost', 'checkpoints')
    checkpoint_callback = xgb.callback.TrainingCheckPoint(
        directory=checkpoint_dir,
        interval=50
    )

    # 6. Initialize XGBoost Regressor
    # Using 'hist' tree method for efficiency on large datasets
    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=10,
        objective='reg:squarederror',
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        reg_lambda=10,
        reg_alpha=1,
        enable_categorical=True,
        callbacks=[checkpoint_callback]
    )
    
    with mlflow.start_run(run_name="XGBoost_Training"):  
        mlflow.log_params({
            "model_type": "xgboost",
            "base_score": 0.5,
            "booster": 'gbtree',
            "tree_method": "hist",
            "n_estimators": n_estimators,
            "early_stopping_rounds": 10,
            "objective": 'reg:squarederror',
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": 42,
            "reg_lambda": 10,
            "reg_alpha": 1,
            "enable_categorical": True
        })

        # 7. Train Model
        print("Fitting XGBoost model...")
        reg.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=True
        )
        
        # 8. Evaluate on Validation Set
        y_val_pred_log = reg.predict(X_val)
        metrics = calculate_metrics(y_val, y_val_pred_log)
        
        print(f"Metrics: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}")
        mlflow.log_metrics(metrics) # Logs all dictionary keys (mae, rmse, bias, smape)
        
        val_score = reg.score(X_val, y_val)
        mlflow.log_metric("val_r2_log_space", val_score)
        
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', reg)
        ])
        
        input_example = X_train_df.head(5)
        
        prediction = full_pipeline.predict(input_example)
        signature = infer_signature(input_example, prediction)
        
        print("Logging Pipeline to MLflow...")
        mlflow.sklearn.log_model(
            full_pipeline, 
            artifact_path="xgboost", 
            signature=signature,
            input_example=input_example
        )
    
        print("Model and metrics logged to MLflow!")
        print(f"Success! Run ID: {mlflow.active_run().info.run_id}")

    # 8. Save Artifacts
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models/xgboost')
    
    # Save Preprocessor (Critical for inference consistency)
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    # Save Model
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(reg, model_path)
    print(f"Model saved to {model_path}")
    
    return reg

if __name__ == "__main__":
    train_xgboost()