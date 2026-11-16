"""
Unit Tests for ChurnIQ Prediction Engine and CLI Interface
Validates inference calculations, input validation, threshold logic,
and CLI validation helpers.
"""

import os
import sys
import pytest
from unittest.mock import patch

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.prediction import get_prediction_engine, ChurnPredictor
from src.cli_predict import prompt_choice, prompt_float, prompt_int


@pytest.fixture(scope="module")
def predictor():
    """Returns the singleton ChurnPredictor engine."""
    return get_prediction_engine()


@pytest.fixture
def valid_high_risk_customer():
    return {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 179.70,
    }


@pytest.fixture
def valid_low_risk_customer():
    return {
        "gender": "Male",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 65,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2925.0,
    }


def test_model_and_preprocessor_loaded(predictor):
    """Verifies that model and preprocessor are loaded and ready in memory."""
    assert predictor._model is not None
    assert predictor._preprocessor is not None


def test_predict_single_high_risk_structure(predictor, valid_high_risk_customer):
    """Verifies output structure and probability ranges for high-risk customer."""
    result = predictor.predict_single(valid_high_risk_customer, threshold=0.50)
    
    assert "churn_probability" in result
    assert "churn_probability_percent" in result
    assert "prediction_result" in result
    assert "is_churn" in result
    assert "risk_level" in result
    assert "explanation" in result
    assert 0.0 <= result["churn_probability"] <= 1.0
    assert result["churn_probability_percent"] == round(result["churn_probability"] * 100, 2)
    assert len(result["explanation"]) > 0


def test_threshold_affects_classification(predictor, valid_high_risk_customer):
    """Ensures threshold parameter dynamically determines the binary classification."""
    # Ultra-low threshold forces churn
    low_thresh_result = predictor.predict_single(valid_high_risk_customer, threshold=0.01)
    assert low_thresh_result["is_churn"] is True
    assert low_thresh_result["prediction_result"] == "High churn probability"

    # Ultra-high threshold forces non-churn
    high_thresh_result = predictor.predict_single(valid_high_risk_customer, threshold=0.99)
    assert high_thresh_result["is_churn"] is False
    assert high_thresh_result["prediction_result"] == "Low churn probability"


def test_predict_single_invalid_inputs(predictor):
    """Verifies input validation rejections for malformed inputs."""
    # Test non-dict input
    with pytest.raises(ValueError, match="Input data must be a JSON object"):
        predictor.predict_single("not a dictionary")

    # Test missing essential fields
    with pytest.raises(ValueError, match="Missing essential field"):
        predictor.predict_single({"tenure": 12})

    # Test out-of-range numerical fields
    with pytest.raises(ValueError, match="must be between"):
        predictor.predict_single({
            "Contract": "One year",
            "tenure": -5,
            "MonthlyCharges": 50.0,
        })

    # Test non-numeric string in numerical field
    with pytest.raises(ValueError, match="must be a valid number"):
        predictor.predict_single({
            "Contract": "One year",
            "tenure": "abc",
            "MonthlyCharges": 50.0,
        })


def test_batch_prediction(predictor, valid_high_risk_customer, valid_low_risk_customer):
    """Verifies batch prediction functionality over multiple customer profiles."""
    customers = [valid_high_risk_customer, valid_low_risk_customer]
    results = predictor.predict_batch(customers, threshold=0.50)
    
    assert len(results) == 2
    assert all("churn_probability" in r for r in results)
    # High risk customer probability should exceed low risk customer probability
    assert results[0]["churn_probability"] > results[1]["churn_probability"]


def test_cli_prompt_helpers():
    """Verifies CLI interactive prompt validation functions."""
    with patch("builtins.input", return_value="Male"):
        choice = prompt_choice("Gender", ["Female", "Male"], "Female")
        assert choice == "Male"

    with patch("builtins.input", return_value="invalid_val"):
        choice = prompt_choice("Gender", ["Female", "Male"], "Female")
        assert choice == "Female"

    with patch("builtins.input", return_value="25.5"):
        flt = prompt_float("Monthly", 0.0, 100.0, 50.0)
        assert flt == 25.5

    with patch("builtins.input", return_value="12"):
        integer_val = prompt_int("Tenure", 0, 72, 6)
        assert integer_val == 12
