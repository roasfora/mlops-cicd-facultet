"""API tests using FastAPI's TestClient (no real server needed)."""
from fastapi.testclient import TestClient

from src.app import app


def test_health_endpoint():
    # `with` runs the startup code, which loads model.joblib
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_valid_class():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"pclass": 1, "sex": "female", "age": 29, "fare": 80.0}
        )
    assert response.status_code == 200
    assert response.json()["survived"] in (0, 1)


def test_predict_accepts_missing_age():
    with TestClient(app) as client:
        response = client.post("/predict", json={"pclass": 3, "sex": "male", "fare": 7.25})
    assert response.status_code == 200


def test_predict_rejects_invalid_input():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"pclass": 9, "sex": "robot", "age": -5, "fare": 10}
        )
    assert response.status_code == 422
