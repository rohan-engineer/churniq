"""
ChurnIQ Command-Line Prediction Interface
Interactive CLI tool allowing users to enter customer information and receive
instant churn probability predictions and risk factors.
Reuses the shared prediction engine without duplicating ML logic.
"""

import sys
import os
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.prediction import get_prediction_engine


def prompt_choice(prompt: str, choices: list, default: str) -> str:
    """Helper to prompt user for a categorical choice."""
    choices_str = "/".join(choices)
    user_val = input(f"{prompt} [{choices_str}] (default: {default}): ").strip()
    if not user_val:
        return default
    for choice in choices:
        if user_val.lower() == choice.lower():
            return choice
    print(f"  Invalid choice '{user_val}'. Using default: '{default}'")
    return default


def prompt_float(prompt: str, min_val: float, max_val: float, default: float) -> float:
    """Helper to prompt user for a float value within bounds."""
    user_val = input(f"{prompt} ({min_val}-{max_val}) [default: {default}]: ").strip()
    if not user_val:
        return default
    try:
        val = float(user_val)
        if min_val <= val <= max_val:
            return val
        print(f"  Value must be between {min_val} and {max_val}. Using default: {default}")
        return default
    except ValueError:
        print(f"  Invalid numeric input. Using default: {default}")
        return default


def prompt_int(prompt: str, min_val: int, max_val: int, default: int) -> int:
    """Helper to prompt user for an integer within bounds."""
    user_val = input(f"{prompt} ({min_val}-{max_val}) [default: {default}]: ").strip()
    if not user_val:
        return default
    try:
        val = int(user_val)
        if min_val <= val <= max_val:
            return val
        print(f"  Value must be between {min_val} and {max_val}. Using default: {default}")
        return default
    except ValueError:
        print(f"  Invalid integer. Using default: {default}")
        return default


def get_interactive_input() -> Dict[str, Any]:
    """Interactively collects all customer features from the terminal."""
    print("\n" + "=" * 60)
    print("      ChurnIQ CLI - Enter Customer Information       ")
    print("=" * 60)

    customer = {}
    
    # Demographics
    print("\n--- Demographics ---")
    customer["gender"] = prompt_choice("Gender", ["Female", "Male"], "Female")
    customer["SeniorCitizen"] = prompt_choice("Senior Citizen", ["0", "1"], "0")
    customer["Partner"] = prompt_choice("Partner", ["Yes", "No"], "No")
    customer["Dependents"] = prompt_choice("Dependents", ["Yes", "No"], "No")
    
    # Account & Contract
    print("\n--- Contract & Account Details ---")
    customer["tenure"] = prompt_int("Tenure (months with service)", 0, 72, 12)
    customer["Contract"] = prompt_choice(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"],
        "Month-to-month"
    )
    customer["PaperlessBilling"] = prompt_choice("Paperless Billing", ["Yes", "No"], "Yes")
    customer["PaymentMethod"] = prompt_choice(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        "Electronic check"
    )
    customer["MonthlyCharges"] = prompt_float("Monthly Charges ($)", 18.0, 150.0, 75.0)
    
    # Total charges estimation if not provided
    default_total = round(customer["tenure"] * customer["MonthlyCharges"], 2)
    customer["TotalCharges"] = prompt_float(
        "Total Charges ($)", 0.0, 15000.0, default_total
    )

    # Telecom Services
    print("\n--- Services ---")
    customer["PhoneService"] = prompt_choice("Phone Service", ["Yes", "No"], "Yes")
    if customer["PhoneService"] == "Yes":
        customer["MultipleLines"] = prompt_choice("Multiple Lines", ["No", "Yes"], "No")
    else:
        customer["MultipleLines"] = "No phone service"

    customer["InternetService"] = prompt_choice(
        "Internet Service", ["DSL", "Fiber optic", "No"], "Fiber optic"
    )
    
    if customer["InternetService"] != "No":
        customer["OnlineSecurity"] = prompt_choice("Online Security", ["Yes", "No"], "No")
        customer["OnlineBackup"] = prompt_choice("Online Backup", ["Yes", "No"], "No")
        customer["DeviceProtection"] = prompt_choice("Device Protection", ["Yes", "No"], "No")
        customer["TechSupport"] = prompt_choice("Tech Support", ["Yes", "No"], "No")
        customer["StreamingTV"] = prompt_choice("Streaming TV", ["Yes", "No"], "Yes")
        customer["StreamingMovies"] = prompt_choice("Streaming Movies", ["Yes", "No"], "Yes")
    else:
        customer["OnlineSecurity"] = "No internet service"
        customer["OnlineBackup"] = "No internet service"
        customer["DeviceProtection"] = "No internet service"
        customer["TechSupport"] = "No internet service"
        customer["StreamingTV"] = "No internet service"
        customer["StreamingMovies"] = "No internet service"

    return customer


def run_prediction_and_display(customer: Dict[str, Any], threshold: float = 0.50):
    """Executes the prediction engine and renders styled output in terminal."""
    predictor = get_prediction_engine()
    result = predictor.predict_single(customer, threshold=threshold)

    print("\n" + "=" * 60)
    print("                ChurnIQ PREDICTION RESULT                  ")
    print("=" * 60)
    print(f" Churn Probability:    {result['churn_probability_percent']}%")
    print(f" Prediction:           {result['prediction_result'].upper()}")
    print(f" Risk Level:           {result['risk_level'].upper()}")
    print(f" Threshold Applied:    {result['threshold_used']}")
    print("-" * 60)
    print(" Key Explanations & Risk Drivers:")
    for factor in result["explanation"]:
        print(f"   * {factor}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ChurnIQ Customer Churn Prediction CLI")
    parser.add_argument("--sample", action="store_true", help="Run with a sample high-risk customer profile")
    parser.add_argument("--threshold", type=float, default=0.50, help="Classification decision threshold (0.0 to 1.0)")
    args = parser.parse_args()

    if args.sample:
        print("\n--> Running prediction on a sample high-risk customer profile...")
        sample_customer = {
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
        run_prediction_and_display(sample_customer, threshold=args.threshold)
    else:
        customer_data = get_interactive_input()
        run_prediction_and_display(customer_data, threshold=args.threshold)


if __name__ == "__main__":
    main()
