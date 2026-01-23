"""
API Database Preparation Module
-------------------------------
Extracts processed data from the training pipeline and populates a 
lightweight SQLite database for real-time inference.
Handles data concatenation, type casting, and index optimization.
"""
import sqlite3
import pandas as pd
import os
from src.utils.data_split import prepare_datasets # Reutiliza a lógica de filtragem e agregação

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
db_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'api_forecast.db')
db_path = os.path.abspath(db_path)

# ==============================================================================
# DATABASE SETUP LOGIC
# ==============================================================================
def setup_api_database():
    """
    Fetches filtered datasets and saves history to SQLite.
    Optimized for Top 10 Brands/Hierarchies as defined in data_split.
    """
    
    # 1. Load Filtered Data
    print("Step 1: Preparing datasets...")
    train, val, test = prepare_datasets()
    full_df = pd.concat([train, val, test]).reset_index(drop=True)

    # 2. Format Data for SQLite
    # SQLite handles strings better for date filtering
    if 'DATE' in full_df.columns:
        full_df['DATE'] = full_df['DATE'].astype(str)
        
    # 3. Establish Connection and Save
    print("Step 2: Connecting to SQLite and saving history...")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # Save only required columns for lag calculation to save space
    full_df[['DATE', 'BRAND', 'PRODUCTHIERARCHY3', 'QUANTITY']].to_sql('history', conn, if_exists='replace', index=False)
    
    # 4. Create Indexes for Performance
    # Essential for fast lookups during API calls
    print("Step 3: Creating indexes...")
    conn.execute('CREATE INDEX idx_pair_date ON history (BRAND, PRODUCTHIERARCHY3, DATE)')
    conn.close()
    
if __name__ == "__main__":
    setup_api_database()