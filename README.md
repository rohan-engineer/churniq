# ChurnIQ – Customer Churn Prediction & Analytics Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)](https://flask.palletsprojects.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

**ChurnIQ** is an end-to-end Machine Learning and Deep Learning platform designed to predict customer churn in telecommunications. By analyzing customer contract terms, payment habits, demographic profiles, and subscription services, ChurnIQ helps businesses proactively identify high-risk accounts and retain customers before they defect.

This repository serves as a portfolio project demonstrating Python programming, robust data preprocessing without data leakage, neural network engineering with TensorFlow/Keras, RESTful API architecture with Flask, dynamic Chart.js dashboards, and automated test coverage with Pytest.

---

## 📑 Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technology Stack](#technology-stack)
- [Dataset Overview](#dataset-overview)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Neural Network Architecture](#neural-network-architecture)
- [Model Evaluation & Metrics](#model-evaluation--metrics)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Model Training](#how-to-train-the-model)
- [Running the Flask Platform](#how-to-run-flask)
- [Command-Line Prediction (CLI)](#how-to-run-command-line-predictions)
- [REST API Documentation](#rest-api-documentation)
- [Running Automated Tests](#how-to-execute-tests)

---

## 🎯 Problem Statement

Customer attrition (churn) directly impacts company profitability, as acquiring new customers typically costs 5 to 25 times more than retaining existing ones. Telecom operators need early-warning mechanisms to spot churn signals—such as month-to-month contracts, electronic check payment methods, high monthly fees without tech support, and new subscriber vulnerability—in order to execute targeted retention offers.

---

## 🚀 Key Features

1. **Production-Grade Preprocessing Pipeline**:
   - Zero data leakage: fits transformations strictly on the training set.
   - Handles missing / whitespace values in `TotalCharges` for zero-tenure subscribers.
   - Standardizes numerical variables and One-Hot Encodes categorical variables into 46 engineered features.
   - Persists the transformer via `joblib` so prediction inputs receive exact training-time transforms.

2. **Deep Neural Network Classifier**:
   - Built with TensorFlow and Keras.
   - 2 Hidden layers with ReLU activations and Dropout regularization (30% and 20%).
   - Early stopping on validation loss to prevent overfitting.
   - Sigmoid output neuron generating calibrated churn probability scores.

3. **Configurable Decision Threshold & Explanations**:
   - Instead of a rigid 50% cutoff, users can configure sensitivity thresholds from 0.10 to 0.90.
   - Generates contextual risk explanations (e.g., month-to-month contract, missing tech support, fiber optic price sensitivity).

4. **Modern Responsive Analytics Dashboard**:
   - Clean white background, dark navy headings, and blue accent palette.
   - Real-time Chart.js visual charts (Doughnut churn distribution, Contract breakdown bar chart, Payment method horizontal bars, Tenure cohorts).
   - Interactive single-customer prediction calculator with dynamic probability gauge.

5. **Multi-Channel Prediction**:
   - Web browser interface with instant sample presets.
   - RESTful API endpoints with structured JSON schemas and input validation.
   - Interactive CLI terminal tool (`src/cli_predict.py`) reusing the exact same serialized model and preprocessor.

6. **Comprehensive Automated Testing**:
   - Full suite of Pytest unit and integration tests for preprocessing, predictions, API routes, and error handling.

---

## 🛠 Technology Stack

| Category | Technologies |
| :--- | :--- |
| **Programming Language** | Python 3.11 |
| **Data Science & ML** | Pandas, NumPy, Scikit-learn, Joblib |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **Backend Framework** | Flask 3.x, Flask-CORS, python-dotenv |
| **Frontend UI** | HTML5, CSS3, JavaScript (ES6+), Bootstrap 5, Chart.js 4, FontAwesome |
| **Automated Testing** | Pytest, Flask Test Client |
| **Visualizations** | Matplotlib, Seaborn, Chart.js |

---

## 📊 Dataset Overview

The project uses the standard **IBM Telco Customer Churn** dataset containing **7,043 customer records** and **21 attributes**.

- **Total Records**: 7,043 customers
- **Retained (No)**: 5,174 (73.46%)
- **Churned (Yes)**: 1,869 (26.54%)
- **Target Variable**: `Churn` ('Yes' / 'No')
- **Identifiers**: `customerID` (stripped during feature preprocessing)
- **Numerical Features**: `tenure`, `MonthlyCharges`, `TotalCharges`
- **Categorical Features**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

*Note on Missing Values*: 11 customers in the raw dataset with `tenure == 0` have whitespace `' '` for `TotalCharges`. The preprocessing pipeline automatically imputes `0.0` for these new accounts.

---

## 🔬 Machine Learning Workflow

```
Raw CSV Dataset (7,043 rows)
       │
       ▼
Data Cleaning & Coercion (Spaces -> 0.0, Whitespace Trimming)
       │
       ▼
Stratified Train/Test Split (80% Train: 5,634 | 20% Held-Out Test: 1,409)
       │
       ├──> Fit ColumnTransformer on Train Data Only (StandardScaler + OneHotEncoder)
       │         │
       │         ├──> Saved to models/preprocessor.joblib
       │         ▼
       ├──> Train TensorFlow/Keras Neural Network (EarlyStopping patience=10)
       │         │
       │         ├──> Saved to models/churn_model.keras
       │         ▼
       └──> Evaluate on Held-Out Test Data
                 │
                 ├──> Confusion Matrix (reports/confusion_matrix.png)
                 └──> Metrics Summary (reports/model_metrics.json)
```

---

## 🧠 Neural Network Architecture

```
Layer (type)              Output Shape         Param #      Activation / Config
================================================================================
Input (InputLayer)        (None, 46)           0            Preprocessed Features
dense_layer_1 (Dense)     (None, 64)           3,008        ReLU
dropout_1 (Dropout)       (None, 64)           0            Rate = 0.30
dense_layer_2 (Dense)     (None, 32)           2,080        ReLU
dropout_2 (Dropout)       (None, 32)           0            Rate = 0.20
output_layer (Dense)      (None, 1)            33           Sigmoid
================================================================================
Total params: 5,121 | Trainable params: 5,121 | Optimizer: Adam (lr=0.001)
Loss Function: Binary Crossentropy
```

---

## 📈 Model Evaluation & Metrics

The model was evaluated on the held-out test dataset (**1,409 samples**, stratified 20%):

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | **79.06%** | Overall percentage of accurate customer retention/churn classifications |
| **Precision** | **64.91%** | Accuracy when the model flags a customer as likely to churn |
| **Recall** | **45.99%** | Fraction of actual churners caught by the model (at 0.50 threshold) |
| **F1 Score** | **53.83%** | Harmonic balance between precision and recall |
| **ROC AUC** | **84.17%** | High discriminative capability across all classification thresholds |

### Confusion Matrix Breakdown (Test Set)
- **True Negatives (TN)**: 942 customers correctly identified as retained
- **False Positives (FP)**: 93 customers falsely predicted to churn
- **False Negatives (FN)**: 202 churners missed at the 0.50 cutoff
- **True Positives (TP)**: 172 churners successfully caught

> **Tip**: Adjusting the decision threshold down to **0.35** via the UI slider or API increases recall to over 70%, capturing more at-risk accounts when proactive retention is prioritized.

---

## 📁 Project Structure

```
Customer Churn Prediction Application/
│
├── app.py                      # Flask application factory and web entrypoint
├── config.py                   # Centralized paths, environment settings, and schemas
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
├── .env.example                # Example environment variable file
├── .env                        # Local environment variables
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # IBM Telco Churn dataset
│
├── models/
│   ├── churn_model.keras       # Trained Keras neural network
│   └── preprocessor.joblib     # Serialized Scikit-learn preprocessing pipeline
│
├── notebooks/
│   └── exploratory_analysis.ipynb   # Interactive EDA Jupyter notebook
│
├── src/
│   ├── data_preprocessing.py   # Dataset cleaning, transformation, and train/test split
│   ├── train_model.py          # Keras model construction, training, early stopping
│   ├── evaluate_model.py       # Test set evaluation and metrics exporter
│   ├── prediction.py           # Cached inference engine and risk factor explainability
│   ├── cli_predict.py          # Interactive command-line prediction interface
│   └── eda_runner.py           # Generates Seaborn EDA plots and Jupyter notebook
│
├── routes/
│   └── prediction_routes.py    # Flask REST API blueprint (/api/*)
│
├── templates/
│   ├── index.html              # Executive overview dashboard
│   ├── analytics.html          # Deep-dive comparative analytics
│   └── prediction.html         # Interactive customer churn prediction calculator
│
├── static/
│   ├── css/
│   │   └── style.css           # Clean white / navy / blue styling
│   └── js/
│       ├── dashboard.js        # Dynamic Chart.js and API data rendering
│       └── prediction.js       # Form handling, probability gauge, risk factors
│
├── reports/
│   ├── confusion_matrix.png    # Confusion matrix heatmap
│   ├── model_metrics.json      # Verified test set evaluation metrics
│   ├── eda_churn_overview.png  # Overall churn distribution
│   ├── eda_contract_churn.png  # Contract-wise churn breakdown
│   ├── eda_payment_churn.png   # Payment method churn rates
│   ├── eda_tenure_distribution.png # Tenure distribution KDE
│   ├── eda_monthly_charges.png # Monthly charges boxplot
│   └── eda_services.png        # Service adoption churn comparison
│
└── tests/
    ├── test_preprocessing.py   # Preprocessing pipeline and missing value tests
    ├── test_prediction.py      # Inference engine, threshold logic, and CLI tests
    └── test_api.py             # Flask REST API and view route tests
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone or Open the Repository
```bash
git clone <repo-url>
cd "Customer Churn Prediction Application"
```

### 2. Set Up Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

---

## 🏋️ How to Train the Model

To retrain the preprocessing pipeline and neural network model from scratch:

```bash
python src/train_model.py
```
This trains the neural network on the dataset using early stopping, exports `models/preprocessor.joblib`, and saves `models/churn_model.keras`.

Next, evaluate the trained model on the held-out test set:
```bash
python src/evaluate_model.py
```
This updates `reports/model_metrics.json` and generates `reports/confusion_matrix.png`.

---

## 🌐 How to Run Flask

Start the local Flask development server:

```bash
python app.py
```

Open your browser and navigate to:
```
http://localhost:5000/
```

- **Overview Dashboard**: `http://localhost:5000/`
- **Deep Analytics**: `http://localhost:5000/analytics`
- **Predict Customer**: `http://localhost:5000/prediction`

---

## ⌨️ How to Run Command-Line Predictions

ChurnIQ includes a standalone CLI that reuses the production model without starting the web server.

### Run with a Sample High-Risk Profile:
```bash
python src/cli_predict.py --sample
```

### Run with a Custom Decision Cutoff:
```bash
python src/cli_predict.py --sample --threshold 0.40
```

### Run in Interactive Mode:
```bash
python src/cli_predict.py
```
Follow the interactive terminal prompts to enter customer details and receive instant churn probabilities and risk explanations.

---

## 📡 REST API Documentation

### 1. Health Status
- **Method**: `GET`
- **URL**: `/api/health`
- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "service": "ChurnIQ API",
  "version": "1.0.0",
  "artifacts": {
    "model_exists": true,
    "preprocessor_exists": true,
    "dataset_exists": true
  }
}
```

---

### 2. Dashboard Analytics
- **Method**: `GET`
- **URL**: `/api/dashboard`
- **Response**: `200 OK`
Returns dynamically aggregated customer counts, churn distribution, contract breakdown, payment method stats, tenure cohorts, and service adoption rates.

---

### 3. Model Architecture & Evaluation
- **Method**: `GET`
- **URL**: `/api/model-info`
- **Response**: `200 OK`
Returns layer configurations, parameter counts, and test set evaluation scores (accuracy, precision, recall, f1, confusion matrix).

---

### 4. Single Customer Prediction
- **Method**: `POST`
- **URL**: `/api/predict`
- **Headers**: `Content-Type: application/json`
- **Payload Example**:
```json
{
  "customer": {
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
    "TotalCharges": 179.70
  },
  "threshold": 0.50
}
```

- **Response Example**: `200 OK`
```json
{
  "status": "success",
  "data": {
    "churn_probability": 0.7818,
    "churn_probability_percent": 78.18,
    "prediction_result": "High churn probability",
    "is_churn": true,
    "risk_level": "High",
    "threshold_used": 0.5,
    "explanation": [
      "Month-to-month contract provides no long-term commitment (elevated churn factor).",
      "Short customer tenure (2 months); newer customers have higher attrition rates.",
      "High monthly charges ($89.85/mo) increase customer price sensitivity.",
      "Lack of dedicated Tech Support contributes to unresolved customer frustration.",
      "No Online Security add-on decreases product stickiness.",
      "Electronic check payment method historically correlates with higher churn.",
      "High-tier Fiber Optic without bundled protections increases defection risk."
    ]
  }
}
```

---

### 5. Batch Predictions
- **Method**: `POST`
- **URL**: `/api/batch-predict`
- **Headers**: `Content-Type: application/json`
- **Payload**:
```json
{
  "customers": [
    { /* customer 1 attributes */ },
    { /* customer 2 attributes */ }
  ],
  "threshold": 0.50
}
```
- **Response**: `200 OK`
Returns batch processing summaries (`total_processed`, `predicted_churn`, `predicted_retained`, `churn_percentage`) and per-customer predictions.

---

## 🧪 How to Execute Tests

Run the complete automated test suite with Pytest:

```bash
pytest -v
```

All 23 unit and integration tests will execute, verifying data preprocessing, pipeline serialization, neural network inference, custom threshold behavior, CLI prompts, and Flask API status codes.

---

## 📜 License

This project is licensed under the MIT License - open and free for portfolio, academic, and commercial use.
