"""
ChurnIQ Flask API Routes
Implements REST endpoints for model health, predictions, batch predictions,
model metadata, and dashboard analytics.
"""

import json
import logging
from typing import Dict, Any
import pandas as pd
from flask import Blueprint, request, jsonify

import config
from src.prediction import get_prediction_engine
from src.data_preprocessing import load_dataset, clean_dataset

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint.
    Reports operational status and verification of ML model and preprocessor artifacts.
    """
    model_loaded = config.MODEL_PATH.exists()
    preprocessor_loaded = config.PREPROCESSOR_PATH.exists()
    dataset_available = config.DATASET_PATH.exists()

    status = "healthy" if (model_loaded and preprocessor_loaded) else "degraded"
    status_code = 200 if status == "healthy" else 503

    return jsonify({
        "status": status,
        "service": "ChurnIQ API",
        "version": "1.0.0",
        "artifacts": {
            "model_exists": model_loaded,
            "preprocessor_exists": preprocessor_loaded,
            "dataset_exists": dataset_available,
        },
    }), status_code


@api_bp.route("/dashboard", methods=["GET"])
def get_dashboard_data():
    """
    Computes and returns comprehensive analytics statistics from the dataset
    for the interactive charts and metrics cards.
    Calculates metrics dynamically without hardcoding.
    """
    try:
        df = load_dataset()
        cleaned_df = clean_dataset(df)

        total_customers = int(len(cleaned_df))
        churn_series = cleaned_df[config.TARGET_COLUMN].str.strip().str.capitalize()
        churn_counts = churn_series.value_counts().to_dict()
        churned_customers = int(churn_counts.get("Yes", 0))
        retained_customers = int(churn_counts.get("No", 0))
        churn_rate = round((churned_customers / total_customers) * 100, 2) if total_customers else 0.0

        # Churn by Contract
        contract_churn = (
            cleaned_df.groupby(["Contract", config.TARGET_COLUMN])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )
        contracts = list(cleaned_df["Contract"].unique())
        contract_data = {
            "labels": contracts,
            "retained": [int(contract_churn.get("No", {}).get(c, 0)) for c in contracts],
            "churned": [int(contract_churn.get("Yes", {}).get(c, 0)) for c in contracts],
        }

        # Churn by Payment Method
        payment_churn = (
            cleaned_df.groupby(["PaymentMethod", config.TARGET_COLUMN])
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )
        payments = list(cleaned_df["PaymentMethod"].unique())
        payment_data = {
            "labels": payments,
            "retained": [int(payment_churn.get("No", {}).get(p, 0)) for p in payments],
            "churned": [int(payment_churn.get("Yes", {}).get(p, 0)) for p in payments],
        }

        # Churn by Tenure Cohorts (0-12, 13-24, 25-36, 37-48, 49-60, 61-72)
        tenure_bins = [0, 12, 24, 36, 48, 60, 72]
        tenure_labels = ["0-12 Mo", "13-24 Mo", "25-36 Mo", "37-48 Mo", "49-60 Mo", "61-72 Mo"]
        cleaned_df["TenureCohort"] = pd.cut(
            cleaned_df["tenure"], bins=tenure_bins, labels=tenure_labels, include_lowest=True
        )
        tenure_churn = (
            cleaned_df.groupby(["TenureCohort", config.TARGET_COLUMN], observed=False)
            .size()
            .unstack(fill_value=0)
            .to_dict()
        )
        tenure_data = {
            "labels": tenure_labels,
            "retained": [int(tenure_churn.get("No", {}).get(t, 0)) for t in tenure_labels],
            "churned": [int(tenure_churn.get("Yes", {}).get(t, 0)) for t in tenure_labels],
        }

        # Monthly Charges distribution summary
        monthly_summary = {
            "mean": round(float(cleaned_df["MonthlyCharges"].mean()), 2),
            "median": round(float(cleaned_df["MonthlyCharges"].median()), 2),
            "min": round(float(cleaned_df["MonthlyCharges"].min()), 2),
            "max": round(float(cleaned_df["MonthlyCharges"].max()), 2),
        }

        # Service adoption counts
        services = ["OnlineSecurity", "TechSupport", "OnlineBackup", "DeviceProtection", "StreamingTV", "StreamingMovies"]
        service_stats = {}
        for s in services:
            if s in cleaned_df.columns:
                active_users = int((cleaned_df[s] == "Yes").sum())
                service_stats[s] = active_users

        return jsonify({
            "status": "success",
            "overview": {
                "total_customers": total_customers,
                "churned_customers": churned_customers,
                "retained_customers": retained_customers,
                "churn_rate_percent": churn_rate,
            },
            "contract_analysis": contract_data,
            "payment_analysis": payment_data,
            "tenure_analysis": tenure_data,
            "monthly_charges_summary": monthly_summary,
            "service_adoption": service_stats,
        }), 200

    except Exception as e:
        logger.error(f"Error computing dashboard data: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@api_bp.route("/predict", methods=["POST"])
def predict_churn():
    """
    Endpoint for individual customer churn predictions.
    Accepts JSON body with customer features and optional decision threshold.
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request payload must be JSON with Content-Type: application/json"}), 400

    payload = request.get_json()
    if not payload:
        return jsonify({"status": "error", "message": "Empty request body"}), 400

    # Extract optional threshold
    threshold = payload.get("threshold", config.DEFAULT_CHURN_THRESHOLD)
    try:
        threshold = float(threshold)
        if not (0.0 <= threshold <= 1.0):
            return jsonify({"status": "error", "message": "Threshold must be between 0.0 and 1.0"}), 400
    except (ValueError, TypeError):
        return jsonify({"status": "error", "message": "Invalid threshold parameter format"}), 400

    # Extract customer dictionary
    customer_data = payload.get("customer", payload)
    if "customer" in payload and not isinstance(payload["customer"], dict):
        return jsonify({"status": "error", "message": "'customer' field must be an object"}), 400

    try:
        predictor = get_prediction_engine()
        result = predictor.predict_single(customer_data, threshold=threshold)
        return jsonify({
            "status": "success",
            "data": result,
        }), 200

    except ValueError as ve:
        return jsonify({"status": "error", "message": str(ve)}), 400
    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 503
    except Exception as e:
        logger.error(f"Prediction inference error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal inference error occurred"}), 500


@api_bp.route("/model-info", methods=["GET"])
def get_model_info():
    """
    Returns neural network architecture summary, hyperparameters,
    and verified evaluation metrics from reports/model_metrics.json.
    """
    metrics = {}
    if config.METRICS_PATH.exists():
        try:
            with open(config.METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read metrics file: {e}")

    model_architecture = {
        "model_type": "Deep Neural Network (Feed-Forward Multilayer Perceptron)",
        "framework": "TensorFlow / Keras",
        "input_features": 46,
        "layers": [
            {"type": "Input", "units": 46, "activation": "None"},
            {"type": "Dense", "units": 64, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.30},
            {"type": "Dense", "units": 32, "activation": "ReLU"},
            {"type": "Dropout", "rate": 0.20},
            {"type": "Dense", "units": 1, "activation": "Sigmoid"},
        ],
        "optimizer": "Adam (lr=0.001)",
        "loss_function": "Binary Crossentropy",
        "evaluation_metrics": metrics,
    }

    return jsonify({
        "status": "success",
        "model_info": model_architecture,
    }), 200


@api_bp.route("/batch-predict", methods=["POST"])
def batch_predict():
    """
    Predicts churn for an array of customer records.
    Payload: {"customers": [...], "threshold": 0.50}
    """
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request payload must be JSON"}), 400

    payload = request.get_json()
    customers = payload.get("customers", [])

    if not isinstance(customers, list) or len(customers) == 0:
        return jsonify({"status": "error", "message": "'customers' must be a non-empty list"}), 400

    threshold = payload.get("threshold", config.DEFAULT_CHURN_THRESHOLD)

    try:
        predictor = get_prediction_engine()
        results = predictor.predict_batch(customers, threshold=threshold)

        total = len(results)
        churn_count = sum(1 for r in results if r.get("is_churn") is True)
        retained_count = total - churn_count

        return jsonify({
            "status": "success",
            "summary": {
                "total_processed": total,
                "predicted_churn": churn_count,
                "predicted_retained": retained_count,
                "churn_percentage": round((churn_count / total) * 100, 2) if total else 0.0,
            },
            "predictions": results,
        }), 200

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
