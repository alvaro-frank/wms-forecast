"""
Unit Tests for Feature Engineering
----------------------------------
Tests the functionality of feature engineering utilities.
Ensures that temporal signals, calendar events, and historical patterns
are correctly generated.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.features import (
    create_features,
    add_weekday_weekend_flags,
    add_lags,
    add_diff
)

# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def sample_df():
    """
    Creates a sample dataframe for testing time-series features.
    Includes a sequence of 10 days with a single product and brand.
    """
    dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'DATE': dates,
        'PRODUCTHIERARCHY3': ['A'] * 10,
        'BRAND': ['B1'] * 10,
        'QUANTITY': np.arange(10, dtype=float) # [0.0, 1.0, ..., 9.0]
    })
    
    df = df.set_index('DATE', drop=False)
    return df

# ==============================================================================
# TIME-BASED FEATURES TESTS
# ==============================================================================

def test_create_features_structure(sample_df):
    """
    Verifies if raw calendar features and cyclical encodings are created.
    """
    df = create_features(sample_df)
    
    expected_cols = [
        'MONTH', 'YEAR', 'DAYOFYEAR', 
        'Week sin', 'Week cos', 
        'Month sin', 'Month cos'
    ]
    
    # Check that all expected columns are in the output dataframe
    for col in expected_cols:
        assert col in df.columns, f"Column {col} missing from output"

def test_cyclical_encoding_values(sample_df):
    """
    Verifies that cyclical features are within the expected range [-1, 1].
    """
    df = create_features(sample_df)
    
    # Check that cyclical features are within the expected range [-1, 1]
    cols_to_check = ['Week sin', 'Week cos', 'Year sin', 'Year cos']
    for col in cols_to_check:
        assert df[col].min() >= -1.0
        assert df[col].max() <= 1.0

# ==============================================================================
# CALENDAR EVENTS TESTS
# ==============================================================================

def test_add_weekday_weekend_flags(sample_df):
    """
    Tests the logic for identifying weekends.
    2023-01-01 was a Sunday.
    """
    df = add_weekday_weekend_flags(sample_df)
    
    # 2023-01-01 is Sunday -> is_weekend must be 1
    assert df.loc['2023-01-01', 'is_weekend'] == 1
    
    # 2023-01-02 is Monday -> is_weekend must be 0   
    assert df.loc['2023-01-02', 'is_weekend'] == 0

# ==============================================================================
# HISTORICAL PATTERNS TESTS (LAGS & DIFFS)
# ==============================================================================

def test_add_lags_logic(sample_df):
    """
    Tests if lag features correctly shift values backwards.
    """
    df = add_lags(sample_df)
    
    # First row lag1 should be NaN, second row lag1 should be 0.0, third row lag1 should be 1.0
    assert np.isnan(df.iloc[0]['lag1'])
    assert df.iloc[1]['lag1'] == 0.0
    assert df.iloc[2]['lag1'] == 1.0

def test_add_diff_logic(sample_df):
    """
    Tests if difference features correctly calculate rate of change.
    """
    df = add_diff(sample_df)
    
    # First row diff1 should be NaN, second row diff1 should be 1.0 (1-0), sixth row diff1 should be 1.0 (5-4)
    assert np.isnan(df.iloc[0]['diff1'])
    assert df.iloc[1]['diff1'] == 1.0
    assert df.iloc[5]['diff1'] == 1.0