"""
Data Handling Utility
---------------------
Provides optimized functions for loading large CSV datasets.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================

def load_data(filepath):
    """
    Loads data from a CSV file efficiently.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded and optimized DataFrame, or None if loading fails.
    """
    try:
        # 1. Read CSV 
        df = pd.read_csv(filepath)

        # 2. Parse Dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
        # 6. Hierarchy lvl 3 Column Renaming
        if 'PRODUCTHIERARCHY' in df.columns:
            df = df.rename(columns={'PRODUCTHIERARCHY': 'PRODUCTHIERARCHY3'})
            df = df.rename(columns={'PRODUCTHIERARCHYNAME': 'PRODUCTHIERARCHY3NAME'})
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None