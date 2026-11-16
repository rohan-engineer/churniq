"""
ChurnIQ Prediction Engine
Provides singleton model loading, preprocessor caching, input validation,
inference execution, and explainable churn risk assessment.
Used by Flask API, CLI, and Web interface without duplicating logic.
"""

import logging
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from src.data_preprocessing import load_preprocessor, transform_single_customer, clean_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Singleton-style predictor class that loads the trained Keras model and
    the Scikit-learn preprocessor once into memory and reuses them across calls.
    """
    _instance = None
    _model = None
    _preprocessor = None

    @classmethod
    def get_instance(cls) -> "ChurnPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.load_artifacts()

    def load_artifacts(self) -> None:
        """
        Loads the preprocessor and neural network model from disk.
        """
        logger.info("Initializing ChurnPredictor and loading serialized artifacts...")
        if not config.PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(
                f"Preprocessor not found at {config.PREPROCESSOR_PATH}. Train the model first."
            )
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {config.MODEL_PATH}. Train the model first."
            )

        self._preprocessor = load_preprocessor(config.PREPROCESSOR_PATH)
        self._model = tf.keras.models.load_model(config.MODEL_PATH)
        logger.info("Artifacts successfully loaded into memory.")

    def validate_customer_input(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validates input fields and types for a single customer prediction.
        
        Args:
            data: Dictionary of customer attributes
            
        Returns:
            Tuple[bool, Optional[str]]: (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Input data must be a JSON object."

        # Verify numerical fields
        numerical_checks = {
            "tenure": (0, 120),
            "MonthlyCharges": (0, 1000),
            "TotalCharges": (0, 20000),
        }
        for field, (min_val, max_val) in numerical_checks.items():
            if field in data:
                try:
                    val = float(data[field])
                    if val < min_val or val > max_val:
                        return False, f"Field '{field}' must be between {min_val} and {max_val}."
                except (ValueError, TypeError):
                    return False, f"Field '{field}' must be a valid number."

        # Check required fields
        required_fields = ["Contract", "tenure", "MonthlyCharges"]
        for rf in required_fields:
            if rf not in data:
                return False, f"Missing essential field: '{rf}'."

        return True, None

    def explain_prediction(self, customer: Dict[str, Any], prob: float) -> List[str]:
        """
        Generates intuitive risk factor explanations based on customer attributes
        and domain knowledge from telecom churn dynamics.
        """
        factors = []
        tenure = float(customer.get("tenure", 0))
        monthly = float(customer.get("MonthlyCharges", 0))
        contract = str(customer.get("Contract", ""))
        tech_support = str(customer.get("TechSupport", ""))
        online_sec = str(customer.get("OnlineSecurity", ""))
        payment = str(customer.get("PaymentMethod", ""))
        internet = str(customer.get("InternetService", ""))

        if contract == "Month-to-month":
            factors.append("Month-to-month contract provides no long-term commitment (elevated churn factor).")
        elif contract in ["One year", "Two year"]:
            factors.append(f"Long-term commitment ({contract}) strongly reduces churn likelihood.")

        if tenure < 12:
            factors.append(f"Short customer tenure ({int(tenure)} months); newer customers have higher attrition rates.")
        elif tenure >= 48:
            factors.append(f"High customer loyalty with {int(tenure)} months of continuous service.")

        if monthly > 75:
            factors.append(f"High monthly charges (${monthly:.2f}/mo) increase customer price sensitivity.")

        if tech_support in ["No", "No internet service"]:
            factors.append("Lack of dedicated Tech Support contributes to unresolved customer frustration.")

        if online_sec in ["No", "No internet service"]:
            factors.append("No Online Security add-on decreases product stickiness.")

        if "Electronic check" in payment:
            factors.append("Electronic check payment method historically correlates with higher churn.")

        if internet == "Fiber optic" and monthly > 80:
            factors.append("High-tier Fiber Optic without bundled protections increases defection risk.")

        if not factors:
            factors.append("Customer exhibits balanced usage metrics without severe churn indicators.")

        return factors

    def predict_single(
        self,
        customer_data: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end churn prediction for a single customer dictionary.
        
        Args:
            customer_data: Dictionary of customer attributes
            threshold: Custom decision threshold (defaults to config.DEFAULT_CHURN_THRESHOLD)
            
        Returns:
            Dict: Comprehensive prediction results
        """
        # Validate input
        is_valid, error = self.validate_customer_input(customer_data)
        if not is_valid:
            raise ValueError(error)

        effective_threshold = (
            float(threshold) if threshold is not None else config.DEFAULT_CHURN_THRESHOLD
        )

        # 1. Transform input features using production preprocessor
        X_input = transform_single_customer(customer_data, self._preprocessor)

        # 2. Model inference
        prob = float(self._model.predict(X_input, verbose=0)[0][0])

        # 3. Decision classification
        is_churn = bool(prob >= effective_threshold)
        prediction_category = (
            "High churn probability" if is_churn else "Low churn probability"
        )
        risk_level = "High" if prob >= 0.65 else ("Moderate" if prob >= 0.40 else "Low")

        # 4. Feature explanations
        reasons = self.explain_prediction(customer_data, prob)

        return {
            "churn_probability": round(prob, 4),
            "churn_probability_percent": round(prob * 100, 2),
            "prediction_result": prediction_category,
            "is_churn": is_churn,
            "risk_level": risk_level,
            "threshold_used": round(effective_threshold, 2),
            "explanation": reasons,
            "customer_summary": {
                "tenure_months": int(customer_data.get("tenure", 0)),
                "contract_type": customer_data.get("Contract", "Unknown"),
                "monthly_charges": float(customer_data.get("MonthlyCharges", 0.0)),
                "total_charges": float(customer_data.get("TotalCharges", 0.0)),
            },
        }

    def predict_batch(
        self,
        customers_list: List[Dict[str, Any]],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes predictions on a list of customer dictionaries.
        """
        results = []
        for cust in customers_list:
            try:
                pred = self.predict_single(cust, threshold=threshold)
                results.append(pred)
            except Exception as e:
                results.append({"error": str(e), "customer": cust})
        return results


def get_prediction_engine() -> ChurnPredictor:
    """Helper to retrieve the shared prediction engine."""
    return ChurnPredictor.get_instance()
