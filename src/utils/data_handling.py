import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """
    Loads data from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        # Tenta converter colunas de data comuns
        # Check for both 'date' and 'DATE' to be robust
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            
        # Rename PRODUCTHIERARCHY column if it exists
        if 'PRODUCTHIERARCHY' in df.columns:
            df = df.rename(columns={'PRODUCTHIERARCHY': 'PRODUCTHIERARCHY3'})
            df = df.rename(columns={'PRODUCTHIERARCHYNAME': 'PRODUCTHIERARCHY3NAME'})
            
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None