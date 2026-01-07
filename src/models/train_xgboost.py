import pandas as pd
import xgboost as xgb
import os
import joblib
from sklearn.metrics import mean_squared_error

# Add src to path if running directly
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

def train_xgboost(processed_data=None):
    """
    Trains an XGBoost model using data from data_split.py
    
    Args:
        processed_data: Optional dictionary with data. If None, loads from data_split.
    """
    print("Preparing data for XGBoost...")
    
    # Load data using the robust split pipeline
    train_data, val_data, test_data = prepare_datasets()
    
    if train_data.empty:
        print("No training data available.")
        return None

    # Define features and target
    FEATURES = train_data.columns.tolist()
    TARGET = 'quantity_next_day'
    
    if TARGET in FEATURES:
        FEATURES.remove(TARGET)
        
    print(f"Training with {len(FEATURES)} features.")
    
    # Check for non-numeric columns remaining
    print("Features dtypes:")
    print(train_data[FEATURES].dtypes)

    X_train = train_data[FEATURES]
    y_train = train_data[TARGET]
    
    X_val = val_data[FEATURES]
    y_val = val_data[TARGET]

    # Initialize XGBoost Regressor with enable_categorical=True
    reg = xgb.XGBRegressor(
        base_score=0.5,
        booster='gbtree',
        tree_method="hist",
        n_estimators=10000,
        early_stopping_rounds=50,
        objective='reg:squarederror',
        max_depth=10,
        learning_rate=0.01,
        random_state=42,
        reg_lambda=10,
        reg_alpha=1,
        enable_categorical=True # Enable support for categorical data
    )

    # Train model
    print("Fitting XGBoost model...")
    reg.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )
    
    # Feature Importance
    feature_importance = pd.DataFrame(
        data=reg.feature_importances_,
        index=reg.feature_names_in_,
        columns=['importance']
    ).sort_values('importance', ascending=False)
    
    print("\nTop 5 Feature Importances:")
    print(feature_importance.head())

    # Save model
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(reg, model_path)
    print(f"Model saved to {model_path}")
    
    return reg

if __name__ == "__main__":
    train_xgboost()