"""
XGBoost Forecast Visualization
------------------------------
Generates plots comparing Actual vs. Predicted values for the XGBoost model.
Capable of plotting a specific Product-Brand pair or automatically generating
plots for all brands within a hierarchy if no brand is specified.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'xgboost_model.joblib')
PREPROCESSOR_PATH = os.path.join(ROOT_DIR, 'models', 'preprocessor.joblib')

# ==============================================================================
# VISUALIZATION LOGIC
# ==============================================================================

def visualize_forecast(hier_code, brand_code):
    """
    Generates and saves a forecast plot for a specific product-brand pair.
    
    Args:
        hier_code (str): The Product Hierarchy ID (e.g., '1060000100001.0').
        brand_code (str): The Brand ID (e.g., '1487.0').
    """
    
    # 1. Load Data
    print("Loading data...")

    _, _, test_data = prepare_datasets()
    
    # 2. Load Model Artifacts
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train it first.")
        return
    
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Preprocessor not found at {PREPROCESSOR_PATH}.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    reg = joblib.load(MODEL_PATH)
    
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    # 3. Define Feature Schema
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2',
                'is_black_friday_week', 
                'is_pre_christmas', 
                'is_post_holiday_slump', 
                'is_payday_zone',
                'days_to_christmas']

    CAT_COLS = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    NUM_COLS = [f for f in features if f not in CAT_COLS]

    # 4. Filter Data for Specific Product/Brand
    mask = (test_data['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)) & \
           (test_data['BRAND'].astype(str) == str(brand_code))

    hier_brand_df = test_data[mask].copy()
    # Ensure categorical columns are strings for OneHotEncoder
    hier_brand_df[CAT_COLS] = hier_brand_df[CAT_COLS].astype(str)
    
    if hier_brand_df.empty:
        print(f"No data found for Hierarchy {hier_code} and Brand {brand_code} in test set.")
        return
    
    # 5. Generate Predictions
    try:
        # Transform features using the loaded preprocessor
        X_sub = preprocessor.transform(hier_brand_df[CAT_COLS + NUM_COLS])
        # Predict
        pred_log = reg.predict(X_sub)
        hier_brand_df['prediction'] = np.expm1(pred_log)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # 6. Prepare Plot Metadata
    # Attempt to retrieve readable names, fall back to "Unknown" if missing
    description = "Unknown Product"
    description = hier_brand_df['PRODUCTHIERARCHY3NAME'].iloc[0]
    
    brand_description = "Unknown Brand"
    brand_description = hier_brand_df['BRANDNAME'].iloc[0]

    # Ensure valid Datetime Index for plotting
    if not isinstance(hier_brand_df.index, pd.DatetimeIndex):
        if 'DATE' in hier_brand_df.columns:
            hier_brand_df = hier_brand_df.set_index('DATE')
            
    hier_brand_df = hier_brand_df.sort_index()
    date_col = hier_brand_df.index

    # 7. Generate Plot
    plt.figure(figsize=(12, 6))
    
    # Actual vs Predicted lines
    plt.plot(date_col, hier_brand_df['quantity_next_day'], label='Actual', alpha=0.7)
    plt.plot(date_col, hier_brand_df['prediction'], label='Predicted', linestyle='--')

    # Titles and Labels
    plt.title(f'XGBoost Forecast\nHierarchy: {hier_code} ({description})\nBrand: {brand_code} ({brand_description})')
    plt.xlabel('Date')
    plt.ylabel('Quantity Next Day')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # 8. Save Plot
    # Sanitize filenames to avoid filesystem errors
    clean_brand = "".join(x for x in brand_description if x.isalnum() or x in " -_").strip().replace(" ", "_")
    clean_hier = "".join(x for x in description if x.isalnum() or x in " -_").strip().replace(" ", "_")
    
    filename = f'xgboost_forecast_{clean_brand}_{clean_hier}.png'
    output_path = os.path.join(ROOT_DIR, 'runs', filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

# ==============================================================================
# SCRIPT ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Test Parameters
    hier_code = '1090000600002.0'
    brand_code = '1791.0'
    
    visualize_forecast(hier_code, brand_code)