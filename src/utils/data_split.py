"""
Data Split & Feature Engineering Pipeline
-----------------------------------------
Prepares the final datasets for model training and evaluation.

Key Steps:
1. Loads raw data via 'data_handling.py'.
2. Aggregates data to ensure unique Day-Product-Brand records.
3. FILTERS data to keep only top performing Brands and Hierarchies (New).
4. Splits data chronologically into Train (70%), Validation (20%), and Test (10%).
5. Generates time-series features (Lags, Diff, EWMA) and calendar flags.
6. Cleans up unused columns to produce lightweight DataFrames.
"""

import pandas as pd
import os
import sys

# Ensure project root is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_handling import load_data
from src.utils.filtering import filter_top_brands_and_hierarchies # <--- NEW IMPORT
from src.utils.features import (
    create_features,
    add_weekday_weekend_flags,
    add_holidays,
    add_calendar_events,
    add_lags,
    add_diff,
    add_ewma,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Define path to data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')

# ==============================================================================
# MAIN PROCESSING FUNCTION
# ==============================================================================

def prepare_datasets():
    """
    Orchestrates the full data preparation pipeline.
    
    Returns:
        tuple: (train_data, val_data, test_data) as pandas DataFrames.
    """
    
    # 1. Load Data
    # --------------------------------------------------------------------------
    df = load_data(DATA_PATH)
    
    # Ensure IDs are strings and Dates are datetime objects
    columns_to_convert = ['PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2', 'BRAND']
    df[columns_to_convert] = df[columns_to_convert].astype(str)
    df['DATE'] = pd.to_datetime(df['DATE'])

    # 2. Aggregation
    # --------------------------------------------------------------------------
    # Ensure only one row per Product-Brand-Date by summing quantities
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    
    if all(col in df.columns for col in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']):
        # Aggregate: Sum Quantity, keep the first occurrence of static columns
        df = df.groupby(
            ['DATE', 'PRODUCTHIERARCHY3', 'BRAND'], as_index=False
        ).agg(
            {col: 'first' for col in df.columns if col not in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']} |
            {'QUANTITY': 'sum'}
        )
    
    # 3. Data Filtering (Top Brands & Hierarchies)
    # --------------------------------------------------------------------------
    # Reduces dataset noise by keeping only the most relevant entities.
    # Logic moved to src/utils/filtering.py for cleanliness.
    print("Filtering Top 10 Brands & Top 10 Hierarchies per Brand...")
    final_filtered_df = filter_top_brands_and_hierarchies(
        df, 
        top_n_brands=10, 
        top_n_hier=10
    )
    
    # 4. Chronological Split
    # --------------------------------------------------------------------------
    final_filtered_df = final_filtered_df.sort_values(by=['DATE'])
    n = len(final_filtered_df)
    
    # 70% Train, 20% Val, 10% Test
    train_data = final_filtered_df[0:int(n*0.7)].copy()
    val_data = final_filtered_df[int(n*0.7):int(n*0.9)].copy()
    test_data = final_filtered_df[int(n*0.9):].copy()

    # 5. Target Creation (Next Day Quantity)
    # --------------------------------------------------------------------------
    train_data['quantity_next_day'] = train_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
    val_data['quantity_next_day'] = val_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
    test_data['quantity_next_day'] = test_data.groupby(groupby_cols)['QUANTITY'].shift(-1)

    # Remove rows where target is NaN (last day of history)
    train_data = train_data.dropna()
    val_data = val_data.dropna()
    test_data = test_data.dropna()

    # 6. Feature Engineering
    # --------------------------------------------------------------------------
    
    # A. Lag Features
    train_data = add_lags(train_data)
    val_data = add_lags(val_data)
    test_data = add_lags(test_data)

    # B. Difference Features
    train_data = add_diff(train_data)
    val_data = add_diff(val_data)
    test_data = add_diff(test_data)

    # C. Moving Averages
    train_data = add_ewma(train_data)
    val_data = add_ewma(val_data)
    test_data = add_ewma(test_data)

    # Set Date Index
    train_data = train_data.set_index('DATE')
    val_data = val_data.set_index('DATE')
    test_data = test_data.set_index('DATE')

    # D. Calendar Features
    train_data = add_holidays(train_data)
    val_data = add_holidays(val_data)
    test_data = add_holidays(test_data)

    train_data = add_weekday_weekend_flags(train_data)
    val_data = add_weekday_weekend_flags(val_data)
    test_data = add_weekday_weekend_flags(test_data)

    # Sort final sets
    train_data = train_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])
    val_data = val_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])
    test_data = test_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])
    
    train_data = add_calendar_events(train_data)
    val_data = add_calendar_events(val_data)
    test_data = add_calendar_events(test_data)

    # E. Cyclic Features
    train_data = create_features(train_data)
    val_data = create_features(val_data)
    test_data = create_features(test_data)

    # 7. Cleanup
    # --------------------------------------------------------------------------
    columns_to_drop = [
        'UNITMEASURE',
        'INITIALDEPOSIT',
        'INITIALPOSITION',
        'FINALDEPOSIT',
        'FINALPOSITON',
        'WEIGHT',
        'LENGTH',
        'WIDTH',
        'HEIGHT',
        'MEASUREMENTUNIT',
        'PRICE',
        'CURRENCY',
        'WEIGHTUNIT',
        'PRODUCT',
        'PRODUCTNAME',
    ]
    
    # Robust drop
    for df_split in [train_data, val_data, test_data]:
        existing_drop = [c for c in columns_to_drop if c in df_split.columns]
        if existing_drop:
            df_split.drop(columns=existing_drop, inplace=True)

    return train_data, val_data, test_data