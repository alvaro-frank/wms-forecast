# Warehouse Management System Product Forecast

A research project for **Time Series Forecasting in Warehouse Management System (WMS)** using **AI-based approaches**.  
It includes models based on **XGBoost** and **LSTM neural networks**, evaluation pipelines, feature engineering utilities, and data visualization tools.

The code was **originally developed in a Jupyter notebook and later adapted into a structured repository.**

![Forecast Plot](runs/forecast.png)

## Features
- Two forecasting models: **XGBoost** and **LSTM** (`src/models/`).
- Automated **feature engineering** and **data preprocessing** (`src/utils/`).
- Comprehensive **data analysis** scripts for exploring warehouse movements (`src/data_analysis/`).
- **Evaluation** modules for both models with metrics and visualizations (`src/eval/`).
- **End-to-end forecasting pipeline** for predicting daily transfer quantities.

## Methodology
The project forecasts **future daily quantities transferred inside a warehouse for each product** using aggregated time series data.  
Each model leverages temporal and statistical features such as:
- **Calendar features:** weekday, month, quarter, year, day of month, day of year.  
- **Lag features:** lag1–lag13 representing previous days’ quantities.  
- **Exponential moving averages (EWMAs):** smoothing trends with spans of 5, 20, and 50 days.  
- **Categorical flags:** weekend and holiday indicators.

The forecasting process follows:
1. **Data preprocessing** → Cleaning and feature generation  
2. **Model training** → XGBoost and LSTM on the same input features  
3. **Evaluation** → MAE, RMSE, MAPE, and visual comparison  
4. **Visualization** → Forecast plots and diagnostic graphs  

## Project Structure
```
src/
  data_analysis/ # Exploratory analysis: monthly, weekly movements, outliers
  eval/ # Evaluation scripts for LSTM & XGBoost
  models/ # Model definitions (lstm.py, xgboost.py)
  utils/ # Data handling, feature generation, visualization helpers
  main.py # Entry point for training and forecasting
  runs/ # Logs and plots
```

## Requirements
Install dependencies via:
```bash
pip install -r src/requirements.txt
```

## Quick Start
Run the main forecasting pipeline:
```bash
python src/main.py
```
This script loads the dataset, generates features, trains both models (XGBoost & LSTM), evaluates their performance, and visualizes forecast results in the `runs/` directory.

## Outputs
- `runs/forecast_*.png` — Forecast visualization (predicted vs. actual)
- `runs/xgboost_results.txt` — XGBoost evaluation metrics
- `runs/lstm_results.txt` — LSTM evaluation metrics
- `runs/model_*_logs.txt` — Model training logs

## Configuration
You can adjust key parameters in:
- `src/utils/features.py` — Feature creation and lags
- `src/models/xgboost.py` - Model architectures and hyperparameters for XGBoost
- `src/models/lstm.py` — Model architectures and hyperparameters for LSTM
- `src/main.py` — Training pipeline, forecasting horizon, and visualization options

## Repro Tips
- Use a fixed random seed for reproducibility.
- Standardize feature sets when comparing models.
- Visualize both predictions and residuals to identify bias or trend drift.
