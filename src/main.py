import sys
import os
import argparse
import pandas as pd

# Adiciona o diretório pai ao sys.path para permitir imports absolutos como 'src.utils...'
# Isto resolve o problema de "ModuleNotFoundError" quando corres o script de dentro de src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Agora podemos importar os módulos do projeto
from src.utils.data_handling import load_data, preprocess_data
from src.models.lstm import train_lstm
from models.train_xgboost import train_xgboost
from src.eval.eval_lstm import evaluate_lstm
from src.eval.eval_xgboost import evaluate_xgboost

def main():
    parser = argparse.ArgumentParser(description="WMS Forecast Pipeline")
    parser.add_argument('--model', type=str, default='lstm', choices=['lstm', 'xgboost'], help='Model to train')
    parser.add_argument('--data_path', type=str, default='data/movements.csv', help='Path to dataset')
    args = parser.parse_args()

    print(f"Loading data from {args.data_path}...")
    # Verifica se o ficheiro existe antes de tentar carregar
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        # Cria dados dummy para teste se não houver CSV real
        print("Generating dummy data for testing purposes...")
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        df = pd.DataFrame({'date': dates, 'quantity': range(100)})
    else:
        df = load_data(args.data_path)

    print("Preprocessing data...")
    # Assumindo que preprocess_data devolve X_train, y_train, X_test, y_test
    # Terás de ajustar conforme a tua implementação real em data_handling.py
    processed_data = preprocess_data(df) 
    
    if args.model == 'lstm':
        print("Training LSTM model...")
        model = train_lstm(processed_data)
        print("Evaluating LSTM...")
        evaluate_lstm(model, processed_data)
        
    elif args.model == 'xgboost':
        print("Training XGBoost model...")
        model = train_xgboost(processed_data)
        print("Evaluating XGBoost...")
        evaluate_xgboost(model, processed_data)

if __name__ == "__main__":
    main()