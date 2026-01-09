"""
Evaluates XGBoost model predictions and computes metrics by product and brand.
- Uses test_data, features, and trained XGBoost pipeline or regressor
- Prints metrics (MAE, RMSE, MAPE, SMAPE, R2, etc.) for each product-brand pair and overall
"""

import pandas as pd
import numpy as np

# Adjust imports to find utils relative to execution path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

def _smape(y_true, y_pred, eps=1e-8):
    # Calculates symmetric mean absolute percentage error
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps))

def _predict_nextday_xgb(df: pd.DataFrame,
                         features: list,
                         cat_cols=('BRAND','PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2'),
                         preprocessor=None,
                         reg=None,
                         xgb_pipeline=None) -> pd.DataFrame:
    
    if 'DATE' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    
    df = df.copy()
    
    # CRITICAL: Convert categories to strings for the OneHotEncoder (matching training)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if 'quantity_next_day' not in df.columns:
        raise ValueError("test_data must include 'quantity_next_day'.")
    y_true = df['quantity_next_day'].to_numpy(dtype=float)

    # Use all columns expected by the preprocessor
    # (The preprocessor selects what it needs by name)
    X = df 

    if xgb_pipeline is not None:
        y_pred = xgb_pipeline.predict(X)
    else:
        if reg is None:
            raise ValueError("Pass either xgb_pipeline OR reg.")
        
        if preprocessor is not None:
            # Transform the data from 32 -> 96 features
            X = preprocessor.transform(X)
            
        y_pred = reg.predict(X)

    out = df[['DATE','BRAND','PRODUCTHIERARCHY3']].copy()
    out['y_true'] = y_true
    out['y_pred'] = y_pred.astype(float)
    return out

def all_pairs_metrics_on_test_xgb(test_data: pd.DataFrame,
                                  features: list,
                                  preprocessor=None,
                                  reg=None,
                                  xgb_pipeline=None) -> pd.DataFrame:
    # Computes metrics for each product-brand pair and overall
    pred_df = _predict_nextday_xgb(
        df=test_data,
        features=features,
        cat_cols=['BRAND', 'PRODUCTHIERARCHY3'], # Explicitly list cat cols
        preprocessor=preprocessor,
        reg=reg,
        xgb_pipeline=xgb_pipeline
    )

    pred_df = pred_df[~pred_df['y_true'].isna()].copy()

    rows = []
    global_y, global_yhat = [], []

    have_brandname = 'BRANDNAME' in test_data.columns
    have_hiername  = 'PRODUCTHIERARCHY3NAME' in test_data.columns

    opt_cols = {}
    if have_brandname:
        opt_cols['BRANDNAME'] = test_data[['BRAND','BRANDNAME']].drop_duplicates().set_index('BRAND')['BRANDNAME']
    if have_hiername:
        opt_cols['PRODUCTHIERARCHY3NAME'] = test_data[['PRODUCTHIERARCHY3','PRODUCTHIERARCHY3NAME']].drop_duplicates().set_index('PRODUCTHIERARCHY3')['PRODUCTHIERARCHY3NAME']

    for (hcode, bcode), g in pred_df.groupby(['PRODUCTHIERARCHY3', 'BRAND'], sort=False, observed=True):
        y = g['y_true'].to_numpy(dtype=float)
        yhat = g['y_pred'].to_numpy(dtype=float)
        n = len(y)
        if n == 0:
            continue

        mae  = float(np.mean(np.abs(yhat - y)))
        rmse = float(np.sqrt(np.mean((yhat - y) ** 2)))
        nz = y != 0
        mape = float(np.mean(np.abs((yhat[nz] - y[nz]) / y[nz]))) * 100 if np.any(nz) else np.nan
        smape = float(_smape(y, yhat)) * 100
        var = np.var(y)
        r2 = float(1.0 - np.sum((yhat - y) ** 2) / (np.sum((y - np.mean(y)) ** 2) + 1e-12)) if var > 0 else np.nan

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

        global_y.append(y)
        global_yhat.append(yhat)

    metrics_df = pd.DataFrame(rows)

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

    base_cols = ["BRAND"] + (["BRANDNAME"] if have_brandname else []) \
              + ["PRODUCTHIERARCHY3"] + (["PRODUCTHIERARCHY3NAME"] if have_hiername else [])
    metric_cols = ["n_points","mae","rmse","mape_%","smape_%","r2",
                   "mean_actual","mean_pred","sum_actual","sum_pred","bias"]
    return metrics_df[base_cols + metric_cols] \
              .sort_values(["BRAND","PRODUCTHIERARCHY3"]) \
              .reset_index(drop=True)

if __name__ == "__main__":
    # Example execution if run directly
    import joblib
    
    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
    MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.joblib')
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')

    if os.path.exists(MODEL_PATH):
        print("Loading test data...")
        _, _, test_data = prepare_datasets()
        
        print(f"Loading preprocessor from {PREPROCESSOR_PATH}")
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        print(f"Loading model from {MODEL_PATH}")
        reg = joblib.load(MODEL_PATH)
        
        features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                    'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                    'is_weekend', 'is_portuguese_holiday', 'lag2', 'lag7', 'lag15',
                    'lag30', 'diff2', 'diff7', 'diff15', 'diff30',
                    'BRAND', 'PRODUCTHIERARCHY3']

        print(f"Evaluating on {len(test_data)} test samples...")
        
        try:
            xgb_metrics = all_pairs_metrics_on_test_xgb(
                test_data=test_data,
                features=features,
                preprocessor=preprocessor,
                reg=reg
            )
            
            # Print specific brand metrics as requested
            target_brand = '1791.0' 
            print(f"\nMetrics for Brand {target_brand}:")
            # Robust filtering
            mask = xgb_metrics['BRAND'].astype(str) == target_brand
            print(xgb_metrics[mask].head(10))
        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Model file not found at {MODEL_PATH}. Run training first.")