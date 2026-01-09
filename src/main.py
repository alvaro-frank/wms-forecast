import sys
import os
import argparse
import joblib

# Adiciona o diretório raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_train(args):
    """Executa o treino do modelo escolhido."""
    if args.model == 'xgboost':
        from src.models.train_xgboost import train_xgboost
        # Passa os argumentos recebidos da linha de comandos para a função
        train_xgboost(
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            n_estimators=args.n_estimators
        )
    elif args.model == 'lstm':
        print("Treino LSTM ainda não configurado com argumentos.")

def run_evaluate(args):
    """Executa a avaliação do modelo."""
    if args.model == 'xgboost':
        from src.utils.data_split import prepare_datasets
        from src.eval.eval_xgboost import all_pairs_metrics_on_test_xgb
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_model.joblib')
        PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

        if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
            print("Erro: Modelo ou Preprocessor não encontrados. Treina primeiro.")
            return

        print("--> Loading Model components...")
        reg = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        print("--> Loading Test Data...")
        _, _, test_data = prepare_datasets()

        features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                    'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                    'is_weekend', 'is_portuguese_holiday', 'lag2', 'lag7', 'lag15',
                    'lag30', 'diff2', 'diff7', 'diff15', 'diff30',
                    'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']

        print("--> Computing Metrics...")
        try:
            if args.model == 'xgboost':
                metrics = all_pairs_metrics_on_test_xgb(
                    test_data=test_data,
                    features=features,
                    preprocessor=preprocessor,
                    reg=reg
                )
                
                if args.brand is None:               
                    print("\n=== Top 5 Melhores Performances (Head) ===")
                    print(metrics.head(5))

                    print("\n=== Top 5 Piores Performances (Tail) ===")
                    print(metrics.tail(5))
                else:
                    # Print specific brand metrics as requested
                    target_brand = args.brand
                    print(f"\nMetrics for Brand {target_brand}:")
                    # Robust filtering
                    mask = metrics['BRAND'].astype(str) == target_brand
                    print(metrics[mask].head(10))
                    
                    print("\nTop 5 Overall Metrics (Tail):")
                    print(metrics.tail(5))
            elif args.model == 'lstm':
                print("Treino LSTM ainda não configurado com argumentos.")
        except Exception as e:
            print(f"Erro na avaliação: {e}")

def run_visualize(args):
    """Gera gráfico de previsão."""
    if args.model == 'xgboost':
        from src.utils.visualization_xgboost import visualize_forecast
        visualize_forecast(args.hierarchy, args.brand)
    elif args.model == 'lstm':
        print("Treino LSTM ainda não configurado com argumentos.")

def main():
    parser = argparse.ArgumentParser(description="WMS Forecast CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Comando: train
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    train_parser.add_argument('--learning_rate', type=float, default=0.01)
    train_parser.add_argument('--max_depth', type=int, default=10)
    train_parser.add_argument('--n_estimators', type=int, default=10000)
    train_parser.set_defaults(func=run_train)

    # Comando: evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate models')
    eval_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    eval_parser.add_argument('--brand', type=str, required=False, default=None)
    eval_parser.set_defaults(func=run_evaluate)

    # Comando: test
    viz_parser = subparsers.add_parser('test', help='Test forecast')
    viz_parser.add_argument('--model', default='xgboost', choices=['xgboost', 'lstm'])
    viz_parser.add_argument('--hierarchy', type=str, required=True)
    viz_parser.add_argument('--brand', type=str, required=True)
    viz_parser.set_defaults(func=run_visualize)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()