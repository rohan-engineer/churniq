"""
ChurnIQ Data Preprocessing Module
Handles loading, cleaning, feature engineering, pipeline creation, and data splitting
for the IBM Telco Customer Churn dataset.
"""

import logging
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Loads the Telco Customer Churn dataset from disk.
    
    Args:
        file_path: Path to the CSV file (defaults to config.DATASET_PATH)
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    path = file_path or config.DATASET_PATH
    logger.info(f"Loading dataset from: {path}")
    
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place {config.DATASET_FILENAME} "
            f"in data/ or run the download utility."
        )
        
    df = pd.read_csv(path)
    logger.info(f"Dataset successfully loaded. Dimensions: {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw customer churn data.
    - Strips whitespace from column names and string values.
    - Coerces 'TotalCharges' into numeric values (spaces become NaN).
    - Fills missing 'TotalCharges' (customers with tenure=0) with 0.0 or median.
    - Converts SeniorCitizen to consistent categorical representation.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    cleaned = df.copy()
    
    # Strip whitespace from string columns
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        
    # TotalCharges in the Telco dataset contains empty strings ' ' for new customers (tenure == 0)
    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        # For customers with tenure == 0, total charges incurred is 0.0
        missing_count = cleaned["TotalCharges"].isna().sum()
        if missing_count > 0:
            logger.info(f"Handling {missing_count} missing values in TotalCharges (imputing 0.0 for zero tenure)")
            cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(0.0)
            
    # Ensure numerical features are properly typed
    for num_col in config.NUMERICAL_FEATURES:
        if num_col in cleaned.columns:
            cleaned[num_col] = pd.to_numeric(cleaned[num_col], errors="coerce").fillna(0.0)
            
    # SeniorCitizen is 0 or 1 in original data, ensure it's treated consistently
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)
        
    return cleaned


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer for preprocessing.
    - StandardScaler for numerical features (tenure, MonthlyCharges, TotalCharges).
    - OneHotEncoder for categorical features (handles unknown categories gracefully).
    
    Returns:
        ColumnTransformer: Unfitted preprocessor
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                config.NUMERICAL_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_training_data(
    df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer]:
    """
    Executes end-to-end preprocessing pipeline for model training:
    1. Cleans dataset.
    2. Separates features and binary target ('Yes' -> 1, 'No' -> 0).
    3. Splits data into train and test sets (stratified to preserve churn ratio).
    4. Fits preprocessor strictly on training data to avoid data leakage.
    5. Transforms both training and test features.
    
    Args:
        df: Raw DataFrame
        test_size: Proportion of dataset to include in test split
        random_state: Seed for reproducibility
        
    Returns:
        Tuple: (X_train, X_test, y_train, y_test, fitted_preprocessor)
    """
    cleaned_df = clean_dataset(df)
    
    # Remove customerID
    if config.ID_COLUMN in cleaned_df.columns:
        logger.info(f"Removing identifier column: {config.ID_COLUMN}")
        
    # Target encoding
    if config.TARGET_COLUMN not in cleaned_df.columns:
        raise ValueError(f"Target column '{config.TARGET_COLUMN}' not found in dataset.")
        
    y_raw = cleaned_df[config.TARGET_COLUMN]
    y = (y_raw.str.lower() == "yes").astype(int).values
    
    # Feature columns
    feature_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    X_raw = cleaned_df[feature_cols]
    
    # Train-test split (stratified by target variable)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train set: {len(X_train_raw)} samples. Test set: {len(X_test_raw)} samples.")
    
    # Preprocessor fitting strictly on training features
    preprocessor = create_preprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    logger.info(f"Transformed feature space dimension: {X_train.shape[1]} features.")
    return X_train, X_test, y_train, y_test, preprocessor


def save_preprocessor(preprocessor: ColumnTransformer, output_path: Optional[str] = None) -> None:
    """
    Serializes fitted preprocessor to disk using joblib.
    """
    path = output_path or config.PREPROCESSOR_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessor serialized and saved to: {path}")


def load_preprocessor(input_path: Optional[str] = None) -> ColumnTransformer:
    """
    Loads fitted preprocessor from disk.
    """
    path = input_path or config.PREPROCESSOR_PATH
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor file not found at: {path}")
    preprocessor = joblib.load(path)
    return preprocessor


def transform_single_customer(
    customer_dict: Dict[str, Any], preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    Transforms a single customer input dictionary into the preprocessed feature array
    ready for neural network inference.
    
    Args:
        customer_dict: Key-value pairs of customer attributes.
        preprocessor: Fitted ColumnTransformer instance.
        
    Returns:
        np.ndarray: 2D array of shape (1, num_features)
    """
    # Create single-row DataFrame
    df_single = pd.DataFrame([customer_dict])
    cleaned_single = clean_dataset(df_single)
    
    # Ensure all required columns exist, imputing defaults if missing
    for col in config.NUMERICAL_FEATURES:
        if col not in cleaned_single.columns:
            cleaned_single[col] = 0.0
            
    for col in config.CATEGORICAL_FEATURES:
        if col not in cleaned_single.columns:
            cleaned_single[col] = "No"
            
    # Filter and order features
    feature_cols = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    X_single = cleaned_single[feature_cols]
    
    # Transform using saved preprocessor
    return preprocessor.transform(X_single)
