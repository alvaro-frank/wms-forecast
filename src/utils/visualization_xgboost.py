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

# Adjust imports to find utils relative to execution path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_handling import load_data
from src.utils.data_split import prepare_datasets

# Define paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'models', 'xgb_qty_model.pkl')

def visualize_forecast(hier_code, brand_code):
    # Load data
    print("Loading data...")
    # final_filtered_df is the raw loaded data
    final_filtered_df = load_data(DATA_PATH)
    
    # test_data comes from the split
    _, _, test_data = prepare_datasets()
    
    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Please train it first.")
        return

    print(f"Loading model from {MODEL_PATH}...")
    reg = joblib.load(MODEL_PATH)

    # Features used for prediction (must match training)
    target = 'quantity_next_day'
    exclude = [target, 'DATE', 'y_true', 'y_pred']
    features = [c for c in test_data.columns if c not in exclude]
    
    CAT_COLS = ['BRAND', 'PRODUCTHIERARCHY3', 'BRANDNAME', 'PRODUCTHIERARCHY3NAME']
    # Filter only available cat cols
    CAT_COLS = [c for c in CAT_COLS if c in test_data.columns]

    def get_hierarchies_for_brand(brand_code, dataframe):
        """
        Returns all product hierarchies for a given brand.
        """
        brand_data = dataframe[dataframe['BRAND'] == brand_code]

        if 'PRODUCTHIERARCHY3NAME' in brand_data.columns:
            hierarchies = brand_data[['PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY3NAME']].drop_duplicates().sort_values(by='PRODUCTHIERARCHY3').reset_index(drop=True)
        else:
            hierarchies = pd.DataFrame({'PRODUCTHIERARCHY3': brand_data['PRODUCTHIERARCHY3'].unique()})
            hierarchies = hierarchies.sort_values(by='PRODUCTHIERARCHY3').reset_index(drop=True)

        return hierarchies
      
    # Filter test data for the selected hierarchy and brand
    # Ensure types match for filtering
    # test_data columns are categorical, convert to string for comparison or cast code
    # Easier to cast dataframe cols to string temporarily for filtering if codes are passed as strings
    
    mask = (test_data['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)) & \
           (test_data['BRAND'].astype(str) == str(brand_code))
           
    hier_brand_df = test_data[mask].copy()
    
    if hier_brand_df.empty:
        print(f"No data found for Hierarchy {hier_code} and Brand {brand_code} in test set.")
        return

    # Ensure categorical columns are 'category' dtype as expected by the model
    # (Data split usually does this, but good to ensure)
    for col in CAT_COLS:
        hier_brand_df[col] = hier_brand_df[col].astype('category')

    # Make predictions
    # If reg is a pipeline it handles everything, if just regressor we pass X
    X_sub = hier_brand_df[features]
    
    # Handle preprocessor if it existed (here assuming direct model or pipeline in 'reg')
    # If you had a separate preprocessor, load it here.
    # Assuming 'reg' loaded from joblib is the full pipeline or capable model.
    try:
        hier_brand_df['prediction'] = reg.predict(X_sub)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # Get description for plot titles
    # Try to get from filtered df, or fall back to raw df if names were dropped in test_data
    description = "Unknown Product"
    if 'PRODUCTHIERARCHY3NAME' in hier_brand_df.columns:
        description = hier_brand_df['PRODUCTHIERARCHY3NAME'].iloc[0]
    elif final_filtered_df is not None and 'PRODUCTHIERARCHY3NAME' in final_filtered_df.columns:
         match = final_filtered_df[final_filtered_df['PRODUCTHIERARCHY3'].astype(str) == str(hier_code)]
         if not match.empty:
             description = match['PRODUCTHIERARCHY3NAME'].iloc[0]

    brand_description = "Unknown Brand"
    if 'BRANDNAME' in hier_brand_df.columns:
        brand_description = hier_brand_df['BRANDNAME'].iloc[0]
    elif final_filtered_df is not None and 'BRANDNAME' in final_filtered_df.columns:
         match = final_filtered_df[final_filtered_df['BRAND'].astype(str) == str(brand_code)]
         if not match.empty:
             brand_description = match['BRANDNAME'].iloc[0]

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