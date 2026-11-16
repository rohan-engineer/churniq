"""
ChurnIQ Configuration Module
Centralizes application settings, file paths, model hyperparameters, and thresholds.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# File paths
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

DATASET_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATASET_PATH = DATA_DIR / DATASET_FILENAME
MODEL_PATH = MODELS_DIR / "churn_model.keras"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"

# Remote dataset source if not present locally
DATASET_URL = (
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

# Application settings
FLASK_ENV = os.getenv("FLASK_ENV", "development")
SECRET_KEY = os.getenv("SECRET_KEY", "churniq-secret-key-default-2026")
PORT = int(os.getenv("PORT", 5000))

# Decision threshold for binary churn classification (0.0 to 1.0)
DEFAULT_CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", 0.50))

# Feature definitions for Telco dataset
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"
