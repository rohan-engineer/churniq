"""
Integration Tests for ChurnIQ Flask REST API and Web Routes
Uses Flask test client to validate HTTP response codes, JSON schemas,
and error handling.
"""

import os
import sys
import json
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app


@pytest.fixture
def client():
    """Provides Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def sample_customer_payload():
    return {
        "customer": {
            "gender": "Female",
            "SeniorCitizen": "0",
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 79.50,
            "TotalCharges": 954.00,
        },
        "threshold": 0.50,
    }


def test_health_endpoint(client):
    """GET /api/health returns 200 and operational artifact statuses."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["artifacts"]["model_exists"] is True
    assert data["artifacts"]["preprocessor_exists"] is True


def test_dashboard_endpoint(client):
    """GET /api/dashboard returns 200 and dynamic aggregate metrics."""
    response = client.get("/api/dashboard")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert "overview" in data
    assert data["overview"]["total_customers"] > 0
    assert "contract_analysis" in data
    assert "payment_analysis" in data
    assert "tenure_analysis" in data


def test_model_info_endpoint(client):
    """GET /api/model-info returns 200, architecture specs and evaluation metrics."""
    response = client.get("/api/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    model_info = data["model_info"]
    assert "layers" in model_info
    assert "evaluation_metrics" in model_info
    assert model_info["evaluation_metrics"]["accuracy"] > 0.70


def test_predict_endpoint_success(client, sample_customer_payload):
    """POST /api/predict with valid payload returns 200 and prediction data."""
    response = client.post(
        "/api/predict",
        data=json.dumps(sample_customer_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    pred = data["data"]
    assert "churn_probability" in pred
    assert "churn_probability_percent" in pred
    assert "prediction_result" in pred
    assert "explanation" in pred


def test_predict_endpoint_flat_payload_compatibility(client, sample_customer_payload):
    """POST /api/predict also accepts flat JSON structure without 'customer' wrapper."""
    flat_payload = sample_customer_payload["customer"].copy()
    flat_payload["threshold"] = 0.55
    response = client.post(
        "/api/predict",
        data=json.dumps(flat_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["data"]["threshold_used"] == 0.55


def test_predict_endpoint_invalid_content_type(client):
    """POST /api/predict rejects non-JSON requests with 400 Bad Request."""
    response = client.post(
        "/api/predict",
        data="plain text body",
        content_type="text/plain",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_predict_endpoint_invalid_threshold(client, sample_customer_payload):
    """POST /api/predict rejects threshold outside [0, 1] with 400."""
    payload = sample_customer_payload.copy()
    payload["threshold"] = 2.5  # Invalid threshold
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "Threshold must be between 0.0 and 1.0" in data["message"]


def test_predict_endpoint_missing_fields(client):
    """POST /api/predict rejects incomplete payloads with 400."""
    payload = {"customer": {"gender": "Male"}}
    response = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert data["status"] == "error"


def test_batch_predict_success(client, sample_customer_payload):
    """POST /api/batch-predict handles array of customers."""
    cust = sample_customer_payload["customer"]
    batch_payload = {
        "customers": [cust, cust],
        "threshold": 0.50,
    }
    response = client.post(
        "/api/batch-predict",
        data=json.dumps(batch_payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "success"
    assert data["summary"]["total_processed"] == 2
    assert len(data["predictions"]) == 2


def test_batch_predict_empty_list_rejected(client):
    """POST /api/batch-predict rejects empty customer list with 400."""
    response = client.post(
        "/api/batch-predict",
        data=json.dumps({"customers": []}),
        content_type="application/json",
    )
    assert response.status_code == 400


def test_frontend_routes(client):
    """Verifies that all frontend HTML views render successfully with 200 OK."""
    for path in ["/", "/analytics", "/prediction"]:
        res = client.get(path)
        assert res.status_code == 200
        assert b"ChurnIQ" in res.data
