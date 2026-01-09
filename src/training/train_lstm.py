import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

# Adiciona diretório raiz para imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.utils.data_split import prepare_datasets

def create_sequences(X, y, time_steps=30):
    """
    Transforma dados 2D em 3D para LSTM: (Amostras, TimeSteps, Features)
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def train_lstm(time_steps=30, epochs=20, batch_size=32, neurons=64, learning_rate=0.001):
    print(f"--> A iniciar treino LSTM: Lookback={time_steps} dias, Neurons={neurons}, LR={learning_rate}")
    
    # 1. Carregar Dados (Usando a mesma split do XGBoost para consistência)
    print("Loading Data...")
    train_df, val_df, test_df = prepare_datasets()

    # 2. Definir Features
    # Usamos as mesmas features temporais e categóricas
    features = ['QUANTITY', 'lag1', 'diff1', 'EWMA_05', 'EWMA_20', 'EWMA_50',
                'Week sin', 'Week cos', 'Month sin', 'Month cos', 'Year sin', 'Year cos',
                'is_weekend', 'is_portuguese_holiday', 
                'BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    
    # Separar Features (X) e Target (y)
    # Importante: Para LSTM, precisamos de garantir que não há NaNs antes de criar sequências
    target_col = 'quantity_next_day'
    
    # Pipeline de Preprocessamento (OneHot + Scaling)
    # LSTMs gostam de dados entre 0 e 1 (ou -1 e 1)
    
    cat_cols = ['BRAND', 'PRODUCTHIERARCHY3', 'PRODUCTHIERARCHY1', 'PRODUCTHIERARCHY2']
    num_cols = [f for f in features if f not in cat_cols]

    # Pipeline Numérica: Imputer -> MinMaxScaler
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", MinMaxScaler())
    ])

    # Pipeline Geral
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols) # sparse=False para LSTM
        ]
    )

    print("Preprocessing & Scaling Data...")
    # Ajustar (Fit) apenas no treino para evitar data leakage
    X_train_processed = preprocessor.fit_transform(train_df[features])
    X_val_processed = preprocessor.transform(val_df[features])
    
    # O target também deve ser escalado para ajudar a convergência (opcional, mas recomendado para MSE)
    target_scaler = MinMaxScaler()
    y_train_scaled = target_scaler.fit_transform(train_df[[target_col]])
    y_val_scaled = target_scaler.transform(val_df[[target_col]])

    # 3. Criar Sequências (Janelas Deslizantes)
    print(f"Creating sequences (Time steps: {time_steps})...")
    X_train_seq, y_train_seq = create_sequences(X_train_processed, y_train_scaled, time_steps)
    X_val_seq, y_val_seq = create_sequences(X_val_processed, y_val_scaled, time_steps)

    print(f"Input Shape: {X_train_seq.shape}") # (Amostras, 30, Features)

    # 4. Construir Modelo LSTM
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
        
        # Camada LSTM
        tf.keras.layers.LSTM(neurons, return_sequences=False, activation='tanh'),
        tf.keras.layers.Dropout(0.2),
        
        # Camada de Saída
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                  loss='mse', 
                  metrics=['mae'])

    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # 5. Treinar
    print("Fitting LSTM model...")
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=[early_stop],
        verbose=1
    )

    # 6. Salvar Artefactos
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Salvar Modelo Keras
    model.save(os.path.join(models_dir, 'lstm_model.keras'))
    print("Model saved to models/lstm_model.keras")

    # Salvar Preprocessor e Target Scaler (Cruciais para inversão depois!)
    joblib.dump(preprocessor, os.path.join(models_dir, 'lstm_preprocessor.joblib'))
    joblib.dump(target_scaler, os.path.join(models_dir, 'lstm_target_scaler.joblib'))
    print("Scalers saved.")

    return model

if __name__ == "__main__":
    train_lstm()