"""
Visualizes XGBoost forecasts for a specific product hierarchy and brand.
- Filters test_data for selected BRAND and PRODUCTHIERARCHY3
- Applies trained XGBoost model to make predictions
- Plots actual vs. predicted next-day quantities
"""


import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Adjust imports to find utils relative to execution path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'xgboost_model.joblib')
PREPROCESSOR_PATH = os.path.join(ROOT_DIR, 'models', 'preprocessor.joblib')

def visualize_forecast(hier_code, brand_code):
    # Load data
    print("Loading data...")

    _, _, test_data = prepare_datasets()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train it first.")
        return
    
    # Load preprocessor
    if not os.path.exists(PREPROCESSOR_PATH):
        print(f"Preprocessor not found at {PREPROCESSOR_PATH}.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    reg = joblib.load(MODEL_PATH)
    
    print(f"Loading preprocessor from {PREPROCESSOR_PATH}...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
            'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
            'is_weekend', 'is_portuguese_holiday', 'lag2', 'lag7', 'lag15',
            'lag30', 'diff2', 'diff7', 'diff15', 'diff30']

    CAT_COLS = ['BRAND', 'PRODUCTHIERARCHY3']
    NUM_COLS = features
    
    mask = (test_data['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)) & \
           (test_data['BRAND'].astype(str) == str(brand_code))

    hier_brand_df = test_data[mask].copy()
    hier_brand_df[CAT_COLS] = hier_brand_df[CAT_COLS].astype(str)
    
    if hier_brand_df.empty:
        print(f"No data found for Hierarchy {hier_code} and Brand {brand_code} in test set.")
        return
    
    # Handle preprocessor if it existed (here assuming direct model or pipeline in 'reg')
    # If you had a separate preprocessor, load it here.
    # Assuming 'reg' loaded from joblib is the full pipeline or capable model.
    try:
        X_sub = preprocessor.transform(hier_brand_df[CAT_COLS + NUM_COLS])
        hier_brand_df['prediction'] = reg.predict(X_sub)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # Get description for plot titles
    # Try to get from filtered df, or fall back to raw df if names were dropped in test_data
    description = "Unknown Product"
    description = hier_brand_df['PRODUCTHIERARCHY3NAME'].iloc[0]
    
    brand_description = "Unknown Brand"
    brand_description = hier_brand_df['BRANDNAME'].iloc[0]

    # Sort by index (date)
    # Ensure index is datetime
    if not isinstance(hier_brand_df.index, pd.DatetimeIndex):
        if 'DATE' in hier_brand_df.columns:
            hier_brand_df = hier_brand_df.set_index('DATE')
            
    hier_brand_df = hier_brand_df.sort_index()
    date_col = hier_brand_df.index

    # Plot actual vs. predicted next-day quantities
    plt.figure(figsize=(12, 6))
    plt.plot(date_col, hier_brand_df['quantity_next_day'], label='Actual', alpha=0.7)
    plt.plot(date_col, hier_brand_df['prediction'], label='Predicted', linestyle='--')

    plt.title(f'XGBoost Forecast\nHierarchy: {hier_code} ({description})\nBrand: {brand_code} ({brand_description})')
    plt.xlabel('Date')
    plt.ylabel('Quantity Next Day')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save or show
    clean_brand = "".join(x for x in brand_description if x.isalnum() or x in " -_").strip().replace(" ", "_")
    clean_hier = "".join(x for x in description if x.isalnum() or x in " -_").strip().replace(" ", "_")
    
    filename = f'xgboost_forecast_{clean_brand}_{clean_hier}.png'
    output_path = os.path.join(ROOT_DIR, 'runs', filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    # plt.show() # Uncomment if running locally with display

if __name__ == "__main__":
    # Parameters provided
    hier_code = '1060000100001.0'
    brand_code = '1487.0'
    
    visualize_forecast(hier_code, brand_code)