"""
Unit Tests for Data Filtering
-----------------------------
Tests the functionality of dataset reduction utilities.
Ensures that only top-performing brands and hierarchies are retained.
"""

import pytest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.filtering import filter_top_brands_and_hierarchies

# ==============================================================================
# FILTERING LOGIC TESTS
# ==============================================================================

def test_filter_top_brands():
    """
    Tests if the function correctly keeps only the top N brands by volume/frequency.
    """
    # Brand A has high volume (10 entries) (QUANTITY=100 each)
    df_a = pd.DataFrame({
        'DATE': pd.date_range('2023-01-01', periods=10),
        'BRAND': 'A',
        'PRODUCTHIERARCHY3': 'H1',
        'QUANTITY': 100
    })
    
    # Brand B has low volume (2 entries) (QUANTITY=1 each)
    df_b = pd.DataFrame({
        'DATE': pd.date_range('2023-01-01', periods=2),
        'BRAND': 'B',
        'PRODUCTHIERARCHY3': 'H1',
        'QUANTITY': 1
    })
    
    df = pd.concat([df_a, df_b])
    
    # Apply filtering to keep only top 1 brand
    filtered_df = filter_top_brands_and_hierarchies(df, top_n_brands=1, top_n_hier=1)
    
    assert 'A' in filtered_df['BRAND'].values
    assert 'B' not in filtered_df['BRAND'].values
    assert len(filtered_df) == 10  # Only Brand A entries remain