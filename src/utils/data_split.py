"""
Splits and processes filtered product movement data for modeling.
- Sorts, splits into train/val/test, and creates next-day targets
- Adds lags, differences, EWMA, holidays, and weekday/weekend flags
- Drops unused columns and prepares final datasets for modeling
"""
import pandas as pd
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_handling import load_data
try:
    from src.utils.features import (
        create_features,
        add_weekday_weekend_flags,
        add_holidays,
        add_lags,
        add_diff,
        add_ewma,
    )
except ImportError:
    # If features.py is not yet populated, defining dummies to prevent crash
    print("Warning: src.utils.features not found or incomplete. Using dummy feature functions.")
    def pass_through(df): return df
    create_features = pass_through
    add_weekday_weekend_flags = pass_through
    add_holidays = pass_through
    add_lags = pass_through
    add_diff = pass_through
    add_ewma = pass_through

# Define path to data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')

def prepare_datasets():
    # Load data
    final_filtered_df = load_data(DATA_PATH)
    
    if final_filtered_df is None:
        print(f"Failed to load data from {DATA_PATH}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Group by DATE, PRODUCTHIERARCHY3, BRAND and sum QUANTITY
    # This aggregates daily sales per product-brand combination
    if all(col in final_filtered_df.columns for col in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']):
        final_filtered_df = final_filtered_df.groupby(
            ['DATE', 'PRODUCTHIERARCHY3', 'BRAND'], as_index=False
        ).agg(
            {col: 'first' for col in final_filtered_df.columns if col not in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']} |
            {'QUANTITY': 'sum'}
        )

    # Sort data by date
    if 'DATE' in final_filtered_df.columns:
        final_filtered_df = final_filtered_df.sort_values(by=['DATE'])
    
    n = len(final_filtered_df)

    # Split data into train, validation, and test sets
    train_data = final_filtered_df[0:int(n*0.7)].copy()
    val_data = final_filtered_df[int(n*0.7):int(n*0.9)].copy()
    test_data = final_filtered_df[int(n*0.9):].copy()

    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    
    # Check if necessary columns are present
    required_cols = groupby_cols + ['QUANTITY']
    missing = [c for c in required_cols if c not in final_filtered_df.columns]
    
    if not missing:
        # Create next-day quantity target for each group
        train_data['quantity_next_day'] = train_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
        val_data['quantity_next_day'] = val_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
        test_data['quantity_next_day'] = test_data.groupby(groupby_cols)['QUANTITY'].shift(-1)

        # Drop rows with missing target
        train_data = train_data.dropna()
        val_data = val_data.dropna()
        test_data = test_data.dropna()
    else:
        print(f"Missing columns {missing} for feature creation. Skipping target generation.")

    # Add features
    train_data = add_lags(train_data)
    val_data = add_lags(val_data)
    test_data = add_lags(test_data)

    train_data = add_diff(train_data)
    val_data = add_diff(val_data)
    test_data = add_diff(test_data)

    train_data = add_ewma(train_data)
    val_data = add_ewma(val_data)
    test_data = add_ewma(test_data)

    # Set DATE as index
    if 'DATE' in train_data.columns:
        train_data = train_data.set_index('DATE')
        val_data = val_data.set_index('DATE')
        test_data = test_data.set_index('DATE')

    train_data = add_holidays(train_data)
    val_data = add_holidays(val_data)
    test_data = add_holidays(test_data)

    train_data = add_weekday_weekend_flags(train_data)
    val_data = add_weekday_weekend_flags(val_data)
    test_data = add_weekday_weekend_flags(test_data)

    # Sort data by date, hierarchy, and brand if columns exist
    sort_cols = ['DATE', 'PRODUCTHIERARCHY3', 'BRAND']
    # 'DATE' might be index now
    if 'DATE' not in train_data.columns and 'DATE' == train_data.index.name:
        train_data = train_data.sort_index()
        val_data = val_data.sort_index()
        test_data = test_data.sort_index()
    elif all(c in train_data.columns for c in sort_cols):
        train_data = train_data.sort_values(sort_cols)
        val_data = val_data.sort_values(sort_cols)
        test_data = test_data.sort_values(sort_cols)

    train_data = create_features(train_data)
    val_data = create_features(val_data)
    test_data = create_features(test_data)

    # Drop unused columns
    columns_to_drop = [
        'UNITMEASURE', 'INITIALDEPOSIT', 'INITIALPOSITION', 'FINALDEPOSIT', 'FINALPOSITON',
        'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY1NAME', 'PRODUCTHIERARCHY2', 'PRODUCTHIERARCHY2NAME',
        # Removed PRODUCTHIERARCHY3NAME and BRANDNAME from drop list so they are kept
        'WEIGHT', 'LENGTH', 'WIDTH', 'HEIGHT', 'MEASUREMENTUNIT', 'PRICE', 'CURRENCY',
        'WEIGHTUNIT', 'PRODUCT', 'PRODUCTNAME', 'DATE' # Ensure DATE column is dropped if present (it's index)
    ]
    
    # Convert categorical columns to category dtype for XGBoost
    # This prevents the ValueError about object columns
    cat_cols = ['PRODUCTHIERARCHY3', 'BRAND', 'PRODUCTHIERARCHY3NAME', 'BRANDNAME']
    for df in [train_data, val_data, test_data]:
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

    existing_drop = [c for c in columns_to_drop if c in train_data.columns]
    train_data = train_data.drop(columns=existing_drop)
    val_data = val_data.drop(columns=existing_drop)
    test_data = test_data.drop(columns=existing_drop)

    return train_data, val_data, test_data

# Load data on import
train_data, val_data, test_data = prepare_datasets()