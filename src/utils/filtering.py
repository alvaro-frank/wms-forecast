"""
Data Filtering Utilities
------------------------
Helper functions to reduce dataset size by selecting only the most 
relevant Brands and Product Hierarchies based on historical volume and frequency.
"""

import pandas as pd

def filter_top_hierarchies_for_brand_group(brand_data_group, top_n_hier=10):
    """
    Selects the top N hierarchies within a specific brand group based on:
    1. Number of unique dates (Data longevity)
    2. Mean quantity (Sales volume)
    
    Args:
        brand_data_group (pd.DataFrame): Data for a single brand.
        top_n_hier (int): Number of top hierarchies to keep.
        
    Returns:
        pd.DataFrame: Filtered dataframe containing only top hierarchies.
    """
    if brand_data_group.empty:
        return pd.DataFrame(columns=brand_data_group.columns)

    # Calculate metrics per hierarchy
    hierarchy_metrics = brand_data_group.groupby('PRODUCTHIERARCHY3').agg(
        num_unique_dates_hier=('DATE', 'nunique'),
        mean_quantity_hier=('QUANTITY', 'mean')
    ).reset_index()

    # Rank and select top N
    top_hierarchies = hierarchy_metrics.sort_values(
        by=['num_unique_dates_hier', 'mean_quantity_hier'],
        ascending=[False, False]
    ).head(top_n_hier)

    top_hierarchy_names = top_hierarchies['PRODUCTHIERARCHY3'].tolist()

    # Filter original data
    return brand_data_group[brand_data_group['PRODUCTHIERARCHY3'].isin(top_hierarchy_names)]

def filter_top_brands_and_hierarchies(df, top_n_brands=10, top_n_hier=10):
    """
    Filters the entire dataset to keep only the top N brands and, 
    within those brands, the top N hierarchies.
    """
    # 1. Identify Top Brands (Longevity + Volume)
    brand_metrics = df.groupby('BRAND').agg(
        num_unique_dates_brand=('DATE', 'nunique'),
        mean_quantity_brand=('QUANTITY', 'mean')
    ).reset_index()

    top_brands_df = brand_metrics.sort_values(
        by=['num_unique_dates_brand', 'mean_quantity_brand'],
        ascending=[False, False]
    ).head(top_n_brands)

    top_brand_names = top_brands_df['BRAND'].tolist()
    
    # 2. Filter Dataset for Top Brands
    df_top_brands = df[df['BRAND'].isin(top_brand_names)].copy()
    
    # 3. Apply Hierarchy Filter per Brand
    # Group by Brand and apply the hierarchy filter function to each group
    final_filtered_df = df_top_brands.groupby('BRAND', group_keys=False).apply(
        lambda x: filter_top_hierarchies_for_brand_group(x, top_n_hier=top_n_hier)
    )
    
    return final_filtered_df.reset_index(drop=True)