# ==============================================================================
# FILE: tests/test_api_wms.py
# DESCRIPTION: Integration Tests for the WMS Forecast API.
#              Verifies endpoints for XGBoost and LSTM predictions, including
#              success cases and error handling using FastAPI TestClient.
# ==============================================================================
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_main_health():
    """
    Verifies if the API documentation (Swagger UI) is reachable.
    """
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_endpoint_success_xgboost():
    """
    Tests a valid XGBoost prediction for a known Brand and Hierarchy.
    Ensures that 'predicted_quantity' and 'plot_path' are present in the response.
    """
    payload = {
        "brand": "1487.0",
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict/xgboost", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quantity" in data
    assert "plot_path" in data
    assert isinstance(data["predicted_quantity"], float)

def test_predict_invalid_brand_xgboost():
    """
    Verifies the API behavior when receiving a non-existent brand for XGBoost.
    Expects a 500 or 404 error depending on implementation.
    """
    payload = {
        "brand": "9999.0",
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict/xgboost", json=payload)

    assert response.status_code == 500
    
def test_predict_endpoint_success_lstm():
    """
    Tests a valid LSTM multi-step prediction for a known Brand and Hierarchy.
    Ensures that 'forecast_horizon' list and 'plot_path' are returned.
    """
    payload = {
        "brand": "1487.0",
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict/lstm", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "forecast_horizon" in data
    assert "plot_path" in data
    assert isinstance(data["forecast_horizon"], list)

def test_predict_invalid_brand_lstm():
    """
    Verifies the API behavior when receiving a non-existent brand for LSTM.
    Expects a 500 or 404 error depending on implementation.
    """
    payload = {
        "brand": "9999.0",
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict/lstm", json=payload)

    assert response.status_code == 500