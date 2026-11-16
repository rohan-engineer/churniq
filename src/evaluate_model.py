"""
ChurnIQ Model Evaluation Module
Evaluates trained model on the held-out test dataset.
Computes Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Generates visualization plots and exports metrics to JSON for the dashboard.
"""

import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
import tensorflow as tf

import config
from src.data_preprocessing import (
    load_dataset,
    prepare_training_data,
    load_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(
    model: Optional[tf.keras.Model] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    threshold: float = config.DEFAULT_CHURN_THRESHOLD,
) -> Dict[str, Any]:
    """
    Evaluates the model on test data and saves metrics and confusion matrix plot.
    If model or test data is not provided, loads saved model and splits dataset freshly
    with the exact same random state to evaluate the held-out set.
    
    Args:
        model: Trained Keras model (optional)
        X_test: Test features (optional)
        y_test: True test labels (optional)
        threshold: Decision threshold for churn classification (default 0.50)
        
    Returns:
        Dict: Dictionary of evaluation metrics
    """
    # 1. Load model if not provided
    if model is None:
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {config.MODEL_PATH}. Train the model first."
            )
        logger.info(f"Loading trained model from {config.MODEL_PATH}")
        model = tf.keras.models.load_model(config.MODEL_PATH)
        
    # 2. Prepare test data if not provided
    if X_test is None or y_test is None:
        logger.info("Loading dataset and extracting held-out test set...")
        df = load_dataset()
        _, X_test, _, y_test, _ = prepare_training_data(df)
        
    # 3. Predict probabilities and apply threshold
    logger.info(f"Running inference on test set ({len(y_test)} samples)...")
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    
    # 4. Calculate actual metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold_used": threshold,
        "test_samples": len(y_test),
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
            "matrix": [[tn, fp], [fn, tp]],
        },
        "classification_report": clf_report,
        "sample_comparisons": [
            {
                "sample_index": i,
                "actual_label": int(y_test[i]),
                "actual_text": "Churn" if y_test[i] == 1 else "Retained",
                "predicted_probability": round(float(y_prob[i]), 4),
                "predicted_label": int(y_pred[i]),
                "predicted_text": "Churn" if y_pred[i] == 1 else "Retained",
                "match": bool(y_test[i] == y_pred[i]),
            }
            for i in range(min(15, len(y_test)))
        ],
    }
    
    # 5. Save metrics JSON
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Evaluation metrics saved to: {config.METRICS_PATH}")
    
    # 6. Plot and save Confusion Matrix visualization
    plot_confusion_matrix(cm, output_path=config.CONFUSION_MATRIX_PATH)
    
    # 7. Print summary
    print("\n" + "=" * 55)
    print("           ChurnIQ MODEL EVALUATION REPORT          ")
    print("=" * 55)
    print(f"Accuracy:    {acc * 100:.2f}%")
    print(f"Precision:   {prec * 100:.2f}%")
    print(f"Recall:      {rec * 100:.2f}%")
    print(f"F1 Score:    {f1 * 100:.2f}%")
    print(f"ROC AUC:     {roc_auc * 100:.2f}%")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"  [TN: {tn:4d} | FP: {fp:4d}]")
    print(f"  [FN: {fn:4d} | TP: {tp:4d}]")
    print("=" * 55 + "\n")
    
    return metrics


def plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    """
    Renders and saves a stylish Confusion Matrix heatmap using Seaborn & Matplotlib.
    """
    plt.figure(figsize=(6, 5))
    labels = ["Retained (No)", "Churned (Yes)"]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("ChurnIQ - Confusion Matrix", fontsize=14, pad=15, fontweight="bold", color="#1a2530")
    plt.xlabel("Predicted Label", fontsize=12, labelpad=10, color="#1a2530")
    plt.ylabel("Actual Label", fontsize=12, labelpad=10, color="#1a2530")
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Confusion matrix plot saved to: {output_path}")


if __name__ == "__main__":
    evaluate_model()
