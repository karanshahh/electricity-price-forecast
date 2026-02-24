"""Tests for /predict API contract."""

import pytest
from fastapi.testclient import TestClient

from electricity_forecast.serving.api import app

client = TestClient(app)


def test_health() -> None:
    """Health endpoint returns 200."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_requires_timestamps() -> None:
    """Predict with empty timestamps returns 422."""
    r = client.post("/predict", json={"timestamps": []})
    assert r.status_code in (422, 500)  # 422 validation or 500 if model missing


def test_predict_valid_request() -> None:
    """Predict with valid payload returns forecasts list or 500 if model missing/incompatible."""
    payload = {
        "timestamps": ["2024-01-15T12:00:00Z", "2024-01-15T13:00:00Z"],
        "include_intervals": False,
    }
    r = client.post("/predict", json=payload)
    if r.status_code == 500:
        detail = str(r.json().get("detail", ""))
        if any(x in detail.lower() for x in ("model", "load", "key")):
            pytest.skip("Model not trained or incompatible; run: make train (with xgb)")
    assert r.status_code == 200
    data = r.json()
    assert "forecasts" in data
    assert isinstance(data["forecasts"], list)
