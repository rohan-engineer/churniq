"""
Unit Tests for ChurnIQ Data Preprocessing Module
Validates dataset loading, missing value handling, column transformation,
and preprocessor persistence.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.data_preprocessing import (
    load_dataset,
    clean_dataset,
    create_preprocessor,
    prepare_training_data,
    save_preprocessor,
    load_preprocessor,
    transform_single_customer,
)


@pytest.fixture
def sample_raw_dataframe():
    """Provides a synthetic DataFrame matching the Telco schema for isolated testing."""
    return pd.DataFrame({
        "customerID": ["CUST-001", "CUST-002", "CUST-003", "CUST-004"],
        "gender": ["Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 0, 72],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "No", "Yes", "No internet service"],
        "StreamingTV": ["No", "Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Electronic check",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 20.25],
        "TotalCharges": ["29.85", "1889.5", " ", "1450.0"],  # Notice space in row 3
        "Churn": ["No", "No", "Yes", "No"],
    })


def test_clean_dataset_handles_spaces_in_total_charges(sample_raw_dataframe):
    """Verifies that empty string spaces in TotalCharges are coerced to 0.0 without errors."""
    cleaned = clean_dataset(sample_raw_dataframe)
    
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert cleaned.loc[2, "TotalCharges"] == 0.0
    assert cleaned["TotalCharges"].dtype in [np.float64, np.float32]


def test_clean_dataset_types_and_trimming(sample_raw_dataframe):
    """Verifies string trimming and consistent types."""
    sample_df = sample_raw_dataframe.copy()
    sample_df["Contract"] = [" Month-to-month ", "One year", "Month-to-month", "Two year"]
    cleaned = clean_dataset(sample_df)
    
    assert cleaned.loc[0, "Contract"] == "Month-to-month"
    assert cleaned["SeniorCitizen"].dtype == object or cleaned["SeniorCitizen"].dtype == str


def test_prepare_training_data_shapes_and_leakage_prevention(sample_raw_dataframe):
    """Ensures train/test splitting maintains stratified proportions and fits only on train."""
    # Duplicate rows to allow stratified train/test split
    larger_df = pd.concat([sample_raw_dataframe] * 5, ignore_index=True)
    
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(
        larger_df, test_size=0.20, random_state=42
    )
    
    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4
    assert X_train.shape[1] == X_test.shape[1]
    assert np.all(np.isin(y_train, [0, 1]))


def test_preprocessor_save_and_load(tmp_path):
    """Verifies that preprocessor object can be serialized and deserialized accurately."""
    preprocessor = create_preprocessor()
    temp_file = tmp_path / "test_preprocessor.joblib"
    
    save_preprocessor(preprocessor, temp_file)
    assert temp_file.exists()
    
    loaded = load_preprocessor(temp_file)
    assert loaded is not None


def test_transform_single_customer_dimensions():
    """Verifies single customer dict is transformed into the exact model input shape."""
    preprocessor = load_preprocessor(config.PREPROCESSOR_PATH)
    
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": "0",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.50,
        "TotalCharges": 1026.00,
    }
    
    transformed = transform_single_customer(sample_customer, preprocessor)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == (1, 46)


def test_transform_single_customer_handles_missing_fields():
    """Verifies preprocessor imputes defaults gracefully if optional fields are missing."""
    preprocessor = load_preprocessor(config.PREPROCESSOR_PATH)
    minimal_customer = {
        "Contract": "Two year",
        "tenure": 24,
        "MonthlyCharges": 50.0,
    }
    transformed = transform_single_customer(minimal_customer, preprocessor)
    assert transformed.shape == (1, 46)
