import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Add src to path if running directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

def train_xgboost(learning_rate=0.01, max_depth=10, n_estimators=10000):
    """
    Trains an XGBoost model using data from data_split.py
    """
    print("Preparing data for XGBoost...")
    
    # Load data using the robust split pipeline
    train_data, val_data, test_data = prepare_datasets()
    
    if train_data.empty:
        print("No training data available.")
        return None

    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
            'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
            'is_weekend', 'is_portuguese_holiday', 'lag2', 'lag7', 'lag15',
            'lag30', 'diff2', 'diff7', 'diff15', 'diff30']

    CAT_COLS = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    NUM_COLS = features

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
            ("num", numeric_pipe, NUM_COLS),
        ],
        sparse_threshold=1.0
    )

    # Build X/y DATAFRAMES that include categoricals + numerics
    X_train_df = train_data[CAT_COLS + NUM_COLS].copy()
    y_train = train_data['quantity_next_day'].values

    X_val_df = val_data[CAT_COLS + NUM_COLS].copy()
    y_val = val_data['quantity_next_day'].values

    X_test_df = test_data[CAT_COLS + NUM_COLS].copy()
    y_test = test_data['quantity_next_day'].values

    # Fit preprocessor on TRAIN only, transform all splits
    X_train = preprocessor.fit_transform(X_train_df)
    X_val = preprocessor.transform(X_val_df)
    X_test = preprocessor.transform(X_test_df)

    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        tree_method="hist",
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        reg_lambda=10,
        reg_alpha=1,
        enable_categorical=True
    )

    # Train model
    print("Fitting XGBoost model...")
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=True
    )

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'training')
    
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
    
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(reg, model_path)
    print(f"Model saved to {model_path}")
    
    return reg

if __name__ == "__main__":
    train_xgboost()