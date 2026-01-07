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
from src.utils.features import (
    create_features,
    add_weekday_weekend_flags,
    add_holidays,
    add_lags,
    add_diff,
    add_ewma,
)

# Define path to data
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'movimentos_saida_mercadoria.csv')

def prepare_datasets():
    # 1. Load Data
    df = load_data(DATA_PATH)
    
    columns_to_convert = ['PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2', 'BRAND']

    df[columns_to_convert] = df[columns_to_convert].astype(str)
    df['DATE'] = pd.to_datetime(df['DATE'])

    # 2. Aggregation (Sum Quantity per Day/Product)
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    if all(col in df.columns for col in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']):
        df = df.groupby(
            ['DATE', 'PRODUCTHIERARCHY3', 'BRAND'], as_index=False
        ).agg(
            {col: 'first' for col in df.columns if col not in ['DATE', 'PRODUCTHIERARCHY3', 'BRAND', 'QUANTITY']} |
            {'QUANTITY': 'sum'}
        )
    
    final_filtered_df = df.sort_values(by=['DATE'])
    n = len(final_filtered_df)
    
    train_data = final_filtered_df[0:int(n*0.7)].copy()
    val_data = final_filtered_df[int(n*0.7):int(n*0.9)].copy()
    test_data = final_filtered_df[int(n*0.9):].copy()

    train_data['quantity_next_day'] = train_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
    val_data['quantity_next_day'] = val_data.groupby(groupby_cols)['QUANTITY'].shift(-1)
    test_data['quantity_next_day'] = test_data.groupby(groupby_cols)['QUANTITY'].shift(-1)

    train_data = train_data.dropna()
    val_data = val_data.dropna()
    test_data = test_data.dropna()

    train_data = add_lags(train_data)
    val_data = add_lags(val_data)
    test_data = add_lags(test_data)

    train_data = add_diff(train_data)
    val_data = add_diff(val_data)
    test_data = add_diff(test_data)

    train_data = add_ewma(train_data)
    val_data = add_ewma(val_data)
    test_data = add_ewma(test_data)

    train_data = train_data.set_index('DATE')
    val_data = val_data.set_index('DATE')
    test_data = test_data.set_index('DATE')

    train_data = add_holidays(train_data)
    val_data = add_holidays(val_data)
    test_data = add_holidays(test_data)

    train_data = add_weekday_weekend_flags(train_data)
    val_data = add_weekday_weekend_flags(val_data)
    test_data = add_weekday_weekend_flags(test_data)

    train_data = train_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])
    val_data = val_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])
    test_data = test_data.sort_values(['DATE', 'PRODUCTHIERARCHY3', 'BRAND'])

    train_data = create_features(train_data)
    val_data = create_features(val_data)
    test_data = create_features(test_data)

    columns_to_drop = [
        'UNITMEASURE',
        'INITIALDEPOSIT',
        'INITIALPOSITION',
        'FINALDEPOSIT',
        'FINALPOSITON',
        'PRODUCTHIERARCHY1',
        'PRODUCTHIERARCHY1NAME',
        'PRODUCTHIERARCHY2',
        'PRODUCTHIERARCHY2NAME',
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

    train_data = train_data.drop(columns=columns_to_drop)
    val_data = val_data.drop(columns=columns_to_drop)
    test_data = test_data.drop(columns=columns_to_drop)

    return train_data, val_data, test_data