"""
Feature engineering utilities for time series forecasting.
- Adds date/time features, cyclical encodings, lags, differences, EWMA, holidays, and weekday/weekend flags
- Used for preparing model-ready datasets
"""

import numpy as np
import pandas as pd
import holidays

def create_features(df):
    """
    Adds date/time and cyclical features to the DataFrame.
    """
    df = df.copy()
    df['DATE'] = df.index
    df['QUARTER'] = df['DATE'].dt.quarter
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['DAYOFYEAR'] = df['DATE'].dt.dayofyear
    df['DAYOFMONTH'] = df['DATE'].dt.day

    timestamp = df.index.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60 
    week = 7 * day 
    month = 30.44 * day 
    year = (365.2425) * day 

    df['Week sin'] = np.sin(timestamp * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp * (2 * np.pi / week))

    df['Month sin'] = np.sin(timestamp * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp * (2 * np.pi / month))

    df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

    return df

def add_weekday_weekend_flags(df):
    """
    Adds weekday and weekend flags to the DataFrame.
    """
    df = df.copy()
    df['WEEKDAY'] = df.index.weekday
    df['is_weekend'] = df['WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def add_holidays(df):
    """
    Adds a flag for Portuguese holidays to the DataFrame.
    """
    portugal_holidays = holidays.Portugal()
    df = df.copy()
    df['is_portuguese_holiday'] = df.index.to_series().map(lambda x: 1 if x in portugal_holidays else 0)
    return df

def add_lags(df):
    """
    Adds lag features for QUANTITY by product hierarchy and brand.
    """
    df = df.copy()
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    df['lag1'] = df.groupby(groupby_cols)['QUANTITY'].shift(1)
    df['lag2'] = df.groupby(groupby_cols)['QUANTITY'].shift(2)
    df['lag7'] = df.groupby(groupby_cols)['QUANTITY'].shift(7)
    df['lag15'] = df.groupby(groupby_cols)['QUANTITY'].shift(15)
    df['lag30'] = df.groupby(groupby_cols)['QUANTITY'].shift(30)
    return df

def add_diff(df):
    """
    Adds difference features for QUANTITY by product hierarchy and brand.
    """
    df = df.copy()
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    df['diff1'] = df.groupby(groupby_cols)['QUANTITY'].diff(1)
    df['diff2'] = df.groupby(groupby_cols)['QUANTITY'].diff(2)
    df['diff7'] = df.groupby(groupby_cols)['QUANTITY'].diff(7)
    df['diff15'] = df.groupby(groupby_cols)['QUANTITY'].diff(15)
    df['diff30'] = df.groupby(groupby_cols)['QUANTITY'].diff(30)
    return df

def add_ewma(df):
    """
    Adds exponentially weighted moving averages for QUANTITY by product hierarchy and brand.
    """
    df = df.copy()
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    df = df.sort_values(['PRODUCTHIERARCHY3', 'BRAND', 'DATE'])

    df['EWMA_05'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EWMA_20'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EWMA_50'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=50, adjust=False).mean())

    return df