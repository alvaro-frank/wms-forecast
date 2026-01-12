"""
XGBoost Evaluation Module
-------------------------
Evaluates the trained XGBoost model on test data.
Computes performance metrics (MAE, RMSE, MAPE, etc.) for each 
Product-Brand pair and aggregates them into a global score.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _smape(y_true, y_pred, eps=1e-8):
    """
    Calculates Symmetric Mean Absolute Percentage Error (SMAPE).
    """
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def _predict_nextday_xgb(df: pd.DataFrame,
                         features: list,
                         cat_cols=('BRAND','PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2'),
                         preprocessor=None,
                         reg=None,
                         xgb_pipeline=None) -> pd.DataFrame:
    """
    Generates predictions for the next day using the XGBoost model.
    Handles data preprocessing (OneHotEncoding) and pipeline execution.
    """
    
    # 1. Handle Indexing
    # Ensure we can access columns even if DATE is the index
    if 'DATE' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    df = df.copy()
    
    # 2. Prepare Categorical Columns
    # CRITICAL: Convert categories to strings to match the OneHotEncoder training state
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # 3. Extract Target
    if 'quantity_next_day' not in df.columns:
        raise ValueError("test_data must include 'quantity_next_day'.")
    y_true = df['quantity_next_day'].to_numpy(dtype=float)

    # 4. Generate Predictions
    X = df 

    if xgb_pipeline is not None:
        y_pred = xgb_pipeline.predict(X)
    else:
        if reg is None:
            raise ValueError("Pass either xgb_pipeline OR reg.")
        
        if preprocessor is not None:
            X = preprocessor.transform(X)
            
        y_pred_log = reg.predict(X)
        y_pred = np.expm1(y_pred_log)

    # 5. Format Output
    out = df[['DATE','BRAND','PRODUCTHIERARCHY3']].copy()
    out['y_true'] = y_true
    out['y_pred'] = y_pred.astype(float)
    return out

# ==============================================================================
# MAIN EVALUATION LOGIC
# ==============================================================================

def all_pairs_metrics_on_test_xgb(test_data: pd.DataFrame,
                                  features: list,
                                  preprocessor=None,
                                  reg=None,
                                  xgb_pipeline=None) -> pd.DataFrame:
    """
    Computes detailed metrics for every Product-Brand pair in the test set.
    Also calculates an 'Overall' global metric row.
    """
    
    # 1. Generate Predictions
    pred_df = _predict_nextday_xgb(
        df=test_data,
        features=features,
        cat_cols=['BRAND', 'PRODUCTHIERARCHY3'], # Explicitly list cat cols
        preprocessor=preprocessor,
        reg=reg,
        xgb_pipeline=xgb_pipeline
    )

    # 2. Clean Data (Remove NaNs)
    pred_df = pred_df[~pred_df['y_true'].isna()].copy()

    rows = []
    global_y, global_yhat = [], []

    # 3. Setup Metadata Lookups (for readable names)
    have_brandname = 'BRANDNAME' in test_data.columns
    have_hiername  = 'PRODUCTHIERARCHY3NAME' in test_data.columns

    opt_cols = {}
    if have_brandname:
        opt_cols['BRANDNAME'] = test_data[['BRAND','BRANDNAME']].drop_duplicates().set_index('BRAND')['BRANDNAME']
    if have_hiername:
        opt_cols['PRODUCTHIERARCHY3NAME'] = test_data[['PRODUCTHIERARCHY3','PRODUCTHIERARCHY3NAME']].drop_duplicates().set_index('PRODUCTHIERARCHY3')['PRODUCTHIERARCHY3NAME']

    # 4. Iterate per Group (Product + Brand)
    for (hcode, bcode), g in pred_df.groupby(['PRODUCTHIERARCHY3', 'BRAND'], sort=False, observed=True):
        y = g['y_true'].to_numpy(dtype=float)
        yhat = g['y_pred'].to_numpy(dtype=float)
        n = len(y)
        if n == 0:
            continue
        
        # Calculate Group Metrics
        # MAE, RMSE
        mae  = float(np.mean(np.abs(yhat - y)))
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        
        # MAPE
        nz = y != 0
        mape = float(np.mean(np.abs((yhat[nz] - y[nz]) / y[nz]))) * 100 if np.any(nz) else np.nan
        
        # SMAPE
        smape = float(_smape(y, yhat)) * 100
        
        # R2
        var = np.var(y)
        r2 = float(1.0 - np.sum((yhat - y) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-12)) if var > 0 else np.nan

        # Append to results
        rows.append({
            "BRAND": bcode,
            **({"BRANDNAME": opt_cols['BRANDNAME'].get(bcode, "")} if have_brandname else {}),
            "PRODUCTHIERARCHY3": hcode,
            **({"PRODUCTHIERARCHY3NAME": opt_cols['PRODUCTHIERARCHY3NAME'].get(hcode, "")} if have_hiername else {}),
            "n_points": int(n),
            "mae": mae,
            "rmse": rmse,
            "mape_%": mape,
            "smape_%": smape,
            "r2": r2,
            "mean_actual": float(np.mean(y)),
            "mean_pred": float(np.mean(yhat)),
            "sum_actual": float(np.sum(y)),
            "sum_pred": float(np.sum(yhat)),
            "bias": float(np.mean(yhat - y)),
        })

        # Collect for global calculation
        global_y.append(y)
        global_yhat.append(yhat)

    metrics_df = pd.DataFrame(rows)
    
    # 5. Calculate Overall Global Metrics
    if global_y:
        Y = np.concatenate(global_y)
        Yh = np.concatenate(global_yhat)
        mae  = float(np.mean(np.abs(Yh - Y)))
        rmse = float(np.sqrt(np.mean((Yh - Y) ** 2)))
        nz = Y != 0
        mape = float(np.mean(np.abs((Yh[nz] - Y[nz]) / Y[nz]))) * 100 if np.any(nz) else np.nan
        smape = float(_smape(Y, Yh)) * 100
        var = np.var(Y)
        r2 = float(1.0 - np.sum((Yh - Y) ** 2) / (np.sum((Y - np.mean(Y)) ** 2) + 1e-12)) if var > 0 else np.nan

        overall = {
            "BRAND": "__ALL__",
            **({"BRANDNAME": ""} if have_brandname else {}),
            "PRODUCTHIERARCHY3": "__ALL__",
            **({"PRODUCTHIERARCHY3NAME": ""} if have_hiername else {}),
            "n_points": int(len(Y)),
            "mae": mae,
            "rmse": rmse,
            "mape_%": mape,
            "smape_%": smape,
            "r2": r2,
            "mean_actual": float(np.mean(Y)),
            "mean_pred": float(np.mean(Yh)),
            "sum_actual": float(np.sum(Y)),
            "sum_pred": float(np.sum(Yh)),
            "bias": float(np.mean(Yh - Y)),
        }
        metrics_df = pd.concat([metrics_df, pd.DataFrame([overall])], ignore_index=True)

    # 6. Format and Sort Columns
    base_cols = ["BRAND"] + (["BRANDNAME"] if have_brandname else []) \
              + ["PRODUCTHIERARCHY3"] + (["PRODUCTHIERARCHY3NAME"] if have_hiername else [])
    metric_cols = ["n_points","mae","rmse","mape_%","smape_%","r2",
                   "mean_actual","mean_pred","sum_actual","sum_pred","bias"]
    return metrics_df[base_cols + metric_cols] \
              .sort_values(["BRAND","PRODUCTHIERARCHY3"]) \
              .reset_index(drop=True)

# ==============================================================================
# SCRIPT EXECUTION (TEST HARNESS)
# ==============================================================================

if __name__ == "__main__":
    import joblib
    
    # 1. Define Paths
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'training'))
    MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')

    if os.path.exists(MODEL_PATH):
        # 2. Load Data and Models
        print("Loading test data...")
        _, _, test_data = prepare_datasets()
        
        print(f"Loading preprocessor from {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        print(f"Loading model from {MODEL_PATH}")
        reg = joblib.load(MODEL_PATH)
        
        # 3. Define Features
        features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2',
                'is_black_friday_week', 
                'is_pre_christmas', 
                'is_post_holiday_slump', 
                'is_payday_zone',
                'days_to_christmas']

        print(f"Evaluating on {len(test_data)} test samples...")
        
        # 4. Run Evaluation
        try:
            xgb_metrics = all_pairs_metrics_on_test_xgb(
                test_data=test_data,
                features=features,
                preprocessor=preprocessor,
                reg=reg
            )
            
            # 5. Output Results
            target_brand = '1791.0' 
            print(f"\nMetrics for Brand {target_brand}:")
            
            mask = xgb_metrics['BRAND'].astype(str) == target_brand
            print(xgb_metrics[mask].head(10))
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file not found at {MODEL_PATH}. Run training first.")