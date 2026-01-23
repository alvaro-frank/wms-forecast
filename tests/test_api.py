import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_read_main_health():
    """Verifica se a API está online (Swagger UI)."""
    response = client.get("/docs")
    assert response.status_code == 200

def test_predict_endpoint_success():
    """Testa uma predição válida para uma marca e hierarquia conhecidas."""
    payload = {
        "brand": "1487.0",
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "predicted_quantity" in data
    assert "plot_path" in data
    assert isinstance(data["predicted_quantity"], float)

def test_predict_invalid_brand():
    """Verifica o comportamento da API para uma marca inexistente."""
    payload = {
        "brand": "9999.0", # Marca que não existe no SQLite
        "hierarchy": "1060000100001.0",
        "date": "2024-12-01"
    }
    response = client.post("/predict", json=payload)
    # Deve retornar erro 500 ou 404 conforme a sua implementação
    assert response.status_code == 500