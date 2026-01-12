"""
Feature Engineering Utilities
-----------------------------
Provides a collection of functions to generate features for time-series forecasting.
Includes transformations for:
- Temporal signals (Cyclical encoding of Week/Month/Year)
- Calendar events (Holidays, Weekends)
- Historical patterns (Lags, Differences, Moving Averages)
"""

import numpy as np
import pandas as pd
import holidays

# ==============================================================================
# TIME-BASED FEATURES (CYCLICAL & RAW)
# ==============================================================================

def create_features(df):
    """
    Generates raw time features (Month, Year) and Cyclical encodings (Sin/Cos).
    
    Cyclical encoding preserves the continuity of time (e.g., Dec to Jan is close, 
    just like 359° to 0°), which raw integers (1 to 12) fail to capture effectively.
    
    Args:
        df (pd.DataFrame): Input dataframe with a DatetimeIndex.
        
    Returns:
        pd.DataFrame: Dataframe with added time features.
    """
    
    df = df.copy()
    
    # Raw Calendar Features
    df['DATE'] = df.index
    df['QUARTER'] = df['DATE'].dt.quarter
    df['MONTH'] = df['DATE'].dt.month
    df['YEAR'] = df['DATE'].dt.year
    df['DAYOFYEAR'] = df['DATE'].dt.dayofyear
    df['DAYOFMONTH'] = df['DATE'].dt.day

    # Cyclical Encoding Setup
    timestamp = df.index.map(pd.Timestamp.timestamp)

    day = 24 * 60 * 60 
    week = 7 * day 
    month = 30.44 * day 
    year = (365.2425) * day 

    # Weekly Cycle (Captures "Day of Week" continuity)
    df['Week sin'] = np.sin(timestamp * (2 * np.pi / week))
    df['Week cos'] = np.cos(timestamp * (2 * np.pi / week))

    # Monthly Cycle (Captures "Day of Month" continuity)
    df['Month sin'] = np.sin(timestamp * (2 * np.pi / month))
    df['Month cos'] = np.cos(timestamp * (2 * np.pi / month))

    # Yearly Cycle (Captures Seasonality)
    df['Year sin'] = np.sin(timestamp * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp * (2 * np.pi / year))

    return df

# ==============================================================================
# CALENDAR EVENTS (FLAGS)
# ==============================================================================

def add_weekday_weekend_flags(df):
    """
    Adds numeric flags for Day of Week and Weekend status.
    
    Returns:
        pd.DataFrame: Dataframe with 'WEEKDAY' (0-6) and 'is_weekend' (0/1).
    """
    df = df.copy()
    df['WEEKDAY'] = df.index.weekday
    # 5 = Saturday, 6 = Sunday
    df['is_weekend'] = df['WEEKDAY'].apply(lambda x: 1 if x >= 5 else 0)
    return df

def add_holidays(df):
    """
    Adds binary flag for Portuguese National Holidays.
    Using the 'holidays' library ensures dynamic calculation based on the year.
    """
    
    portugal_holidays = holidays.Portugal()
    df = df.copy()
    
    # Check if the date index exists in the holiday calendar
    df['is_portuguese_holiday'] = df.index.to_series().map(lambda x: 1 if x in portugal_holidays else 0)
    return df

def add_calendar_events(df):
    """
    Adds manual event features to simulate promotions and seasonality.
    """
    if 'DATE' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    # Garantir datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # 1. Black Friday (Aprox. última sexta de Novembro)
    # Normalmente o pico dura a "Black Week" inteira
    # Vamos marcar a última semana de Novembro como "Alta Probabilidade de Promo"
    df['is_black_friday_week'] = df['DATE'].apply(
        lambda x: 1 if (x.month == 11 and x.day >= 23) else 0
    )

    # 2. Época de Natal (1 a 23 de Dezembro)
    df['is_pre_christmas'] = df['DATE'].apply(
        lambda x: 1 if (x.month == 12 and 1 <= x.day <= 23) else 0
    )

    # 3. "Ressaca" de Ano Novo (Janeiro fraco)
    df['is_post_holiday_slump'] = df['DATE'].apply(
        lambda x: 1 if (x.month == 1 and x.day <= 15) else 0
    )

    # 4. Payday Effect (Dias 28, 29, 30, 31 e dia 1)
    # As pessoas compram mais quando recebem o salário
    df['is_payday_zone'] = df['DATE'].apply(
        lambda x: 1 if (x.day >= 28 or x.day == 1) else 0
    )
    
    # 5. Contagem Regressiva para o Natal (Feature contínua muito forte)
    # Ajuda o modelo a entender a "urgência" de compra
    def get_days_to_xmas(d):
        if d.month == 12 and d.day <= 25:
            return 25 - d.day
        return 0 # Fora de dezembro ignoramos
        
    df['days_to_christmas'] = df['DATE'].apply(get_days_to_xmas)

    # Voltar a por o indice se necessário
    if 'DATE' in df.columns:
        df = df.set_index('DATE', drop=False)
        
    return df

# ==============================================================================
# HISTORICAL FEATURES (LAGS, DIFFS, MOVING AVERAGES)
# ==============================================================================

def add_lags(df):
    """
    Generates Lag features (Past values) for specific time windows.
    Crucial for Autoregressive models (AR) to learn from recent history.
    
    Lags: 1, 2, 7 (Weekly), 15, 30 (Monthly)
    """
    
    df = df.copy()
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    
    # Shift data backwards by N days within each Product-Brand group
    df['lag1'] = df.groupby(groupby_cols)['QUANTITY'].shift(1)
    df['lag2'] = df.groupby(groupby_cols)['QUANTITY'].shift(2)
    df['lag7'] = df.groupby(groupby_cols)['QUANTITY'].shift(7)
    df['lag15'] = df.groupby(groupby_cols)['QUANTITY'].shift(15)
    df['lag30'] = df.groupby(groupby_cols)['QUANTITY'].shift(30)
    return df

def add_diff(df):
    """
    Generates Difference features (Velocity/Trend) to make data stationary.
    Captures the rate of change between time steps.
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
    Adds Exponentially Weighted Moving Averages (EWMA).
    Smoothes out noise while giving more weight to recent observations.
    
    Spans: 
    - 5 (Short-term trend)
    - 20 (Monthly trend)
    - 50 (Quarterly trend)
    """
    
    df = df.copy()
    groupby_cols = ["PRODUCTHIERARCHY3", "BRAND"]
    
    # Sort to ensure rolling window calculation is chronologically correct
    df = df.sort_values(['PRODUCTHIERARCHY3', 'BRAND', 'DATE'])
    
    df['EWMA_05'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=5, adjust=False).mean())
    df['EWMA_20'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    df['EWMA_50'] = df.groupby(groupby_cols)['QUANTITY'].transform(lambda x: x.ewm(span=50, adjust=False).mean())

    return df