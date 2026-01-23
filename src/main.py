"""
WMS Forecast CLI Entry Point
----------------------------
Main command-line interface for the WMS Forecasting project.
Handles argument parsing and dispatching for:
- Training (XGBoost & LSTM)
- Evaluation
- Visualization/Testing
"""

import sys
import os
import argparse
import joblib

# Ensure src is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ==============================================================================
# COMMAND HANDLERS
# ==============================================================================

def run_train(args):
    """
    Executes the training pipeline for the selected model.
    """
    if args.model == 'xgboost':
        from src.training.train_xgboost import train_xgboost

        train_xgboost(
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators,
            resume=args.resume
        )
    elif args.model == 'lstm':
        from src.training.train_lstm import train_lstm
        train_lstm(
            time_steps=args.time_steps,
            forecast_horizon=args.forecast_horizon,
            epochs=args.epochs,
            batch_size=args.batch_size,
            neurons=args.neurons,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            resume=args.resume
        )

def run_evaluate(args):
    """
    Executes the evaluation pipeline.
    """
    
    # 1. Load Data
    from src.utils.data_split import prepare_datasets
    print("Loading test data")
    _, _, test_data = prepare_datasets() 
    
    if args.model == 'xgboost':
        from src.eval.eval_xgboost import all_pairs_metrics_on_test_xgb
        MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models/xgboost')
        MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_final_model.joblib')
        PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'preprocessor.joblib')
        
        if not os.path.exists(MODEL_PATH):
            print("Model not found. Run 'train' first.")
            return

        print("Loading model & preprocessor...")
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        reg = joblib.load(MODEL_PATH)
        
        features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                    'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                    'is_weekend', 'is_portuguese_holiday', 'lag2', 'lag7', 'lag15',
                    'lag30', 'diff2', 'diff7', 'diff15', 'diff30',
                    'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']

        print("Calculating metrics...")
        metrics = all_pairs_metrics_on_test_xgb(test_data, features, preprocessor=preprocessor, reg=reg)
        
        if args.brand is None:               
            print("\n=== Top 5 Best Performances (Head) ===")
            print(metrics.head(5))

            print("\n=== Top 5 Worst Performances (Tail) ===")
            print(metrics.tail(5))
        else:
            print(f"\n--- Metrics for Brand {args.brand} ---")
            print(metrics[metrics['BRAND'] == args.brand])

    elif args.model == 'lstm':
        from src.eval.eval_lstm import evaluate_lstm
        
        # Try to load metadata if it exists
        meta_path = os.path.join(os.path.dirname(__file__), '..', 'models/lstm', 'lstm_metadata.joblib')
        time_steps = 30
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            time_steps = meta.get('time_steps', 30)
            print(f"Loaded time_steps={time_steps} from metadata.")
        
        metrics = evaluate_lstm(test_data, time_steps=time_steps)
        
        if metrics is not None:
            if args.brand is None:               
                print("\n=== Top 5 Best Performances (Head) ===")
                print(metrics.head(5))

                print("\n=== Top 5 Worst Performances (Tail) ===")
                print(metrics.tail(5))
            else:
                print(f"\n--- Metrics for Brand {args.brand} ---")
                print(metrics[metrics['BRAND'] == args.brand])

def run_visualize(args):
    """
    Executes the visualization/testing pipeline.
    """
    if args.model == 'xgboost':
        from src.utils.visualization_xgboost import visualize_forecast
        visualize_forecast(args.hierarchy, args.brand)
    elif args.model == 'lstm':
        from src.utils.visualization_lstm import visualize_forecast_lstm
        visualize_forecast_lstm(args.hierarchy, args.brand)

def main():
    parser = argparse.ArgumentParser(description="WMS Forecast CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- TRAIN COMMAND ---
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    train_parser.add_argument('--resume', action='store_true', help='Resume training from existing checkpoint')
    
    train_parser.add_argument('--learning_rate', type=float, default=0.01)
    
    # Args XGBoost
    train_parser.add_argument('--max_depth', type=int, default=10)
    train_parser.add_argument('--n_estimators', type=int, default=10000)
    
    # Args LSTM
    train_parser.add_argument('--time_steps', type=int, default=30)
    train_parser.add_argument('--forecast_horizon', type=int, default=7)
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch_size', type=int, default=32)
    train_parser.add_argument('--neurons', type=int, default=64)
    train_parser.add_argument('--dropout', type=float, default=0.2)
    
    train_parser.set_defaults(func=run_train)

    # --- EVALUATE COMMAND ---
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    eval_parser.add_argument('--brand', type=str, required=False, default=None)
    eval_parser.set_defaults(func=run_evaluate)

    # --- TEST COMMAND ---
    viz_parser = subparsers.add_parser('test', help='Test forecast')
    viz_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    viz_parser.add_argument('--hierarchy', type=str, required=True)
    viz_parser.add_argument('--brand', type=str, required=True)
    viz_parser.set_defaults(func=run_visualize)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()