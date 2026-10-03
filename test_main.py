import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_predict_positive(client):
    response = client.post(
        "/predict",
        json={"text": "Заказ пришёл быстро, качество отличное, продавцу спасибо"},
    )
    assert response.status_code == 200
    assert response.json()["label"] == "positive"


def test_predict_negative(client):
    response = client.post(
        "/predict",
        json={"text": "Товар бракованный, деньги не вернули, продавец не отвечает"},
    )
    assert response.status_code == 200
    assert response.json()["label"] == "negative"


def test_response_structure(client):
    response = client.post("/predict", json={"text": "Товар отправлен"})
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"label", "score", "probabilities"}
    assert data["label"] in {"negative", "neutral", "positive"}
    assert 0.0 <= data["score"] <= 1.0
    assert set(data["probabilities"].keys()) == {"negative", "neutral", "positive"}


def test_probabilities_sum_to_one(client):
    response = client.post("/predict", json={"text": "Доставка заняла пять дней"})
    probabilities = response.json()["probabilities"]
    assert abs(sum(probabilities.values()) - 1.0) < 0.01


def test_empty_text_rejected(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_whitespace_text_rejected(client):
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_missing_field_rejected(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_wrong_type_rejected(client):
    response = client.post("/predict", json={"text": 12345})
    assert response.status_code == 422


def test_too_long_text_rejected(client):
    response = client.post("/predict", json={"text": "а" * 5001})
    assert response.status_code == 422


def test_unknown_endpoint(client):
    response = client.get("/unknown")
    assert response.status_code == 404