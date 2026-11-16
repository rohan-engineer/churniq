"""
ChurnIQ Model Training Module
Constructs, trains, and saves a Deep Learning Neural Network classifier using TensorFlow & Keras.
Uses early stopping and validation splitting to prevent overfitting.
"""

import logging
import os
import sys
from typing import Tuple, Dict, Any, Optional
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

import config
from src.data_preprocessing import (
    load_dataset,
    prepare_training_data,
    save_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_neural_network(input_dim: int, learning_rate: float = 0.001) -> keras.Model:
    """
    Builds the Sequential Neural Network model according to specifications:
    - Input layer matching preprocessed feature dimensions.
    - Dense hidden layer (64 units) + ReLU activation.
    - Dropout layer (0.30) to prevent overfitting.
    - Dense hidden layer (32 units) + ReLU activation.
    - Dropout layer (0.20).
    - Dense output layer (1 unit) + Sigmoid activation for binary probability output.
    
    Args:
        input_dim: Number of input features
        learning_rate: Initial Adam optimizer learning rate
        
    Returns:
        keras.Model: Compiled Keras model
    """
    logger.info(f"Building neural network model with input dimension: {input_dim}")
    
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,), name="input_features"),
            layers.Dense(64, activation="relu", name="dense_layer_1"),
            layers.Dropout(0.3, name="dropout_1"),
            layers.Dense(32, activation="relu", name="dense_layer_2"),
            layers.Dropout(0.2, name="dropout_2"),
            layers.Dense(1, activation="sigmoid", name="output_layer"),
        ],
        name="ChurnIQ_Neural_Network",
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    
    model.summary(print_fn=logger.info)
    return model


def train_model(
    epochs: int = 60,
    batch_size: int = 32,
    patience: int = 10,
    learning_rate: float = 0.001,
) -> Tuple[keras.Model, Any, np.ndarray, np.ndarray]:
    """
    Loads data, trains preprocessor, builds neural network, and fits the model.
    Saves both the preprocessor and the trained model.
    
    Returns:
        Tuple: (trained_model, preprocessor, X_test, y_test)
    """
    # 1. Load dataset
    df = load_dataset()
    
    # 2. Preprocess data and split into train/test sets
    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(df)
    
    # 3. Save preprocessor for production inference
    save_preprocessor(preprocessor)
    
    # 4. Build neural network
    input_dim = X_train.shape[1]
    model = build_neural_network(input_dim=input_dim, learning_rate=learning_rate)
    
    # 5. Configure callbacks: Early Stopping prevents overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    
    # 6. Train model with validation split
    logger.info("Starting model training with early stopping...")
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )
    
    # 7. Save model to disk
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(config.MODEL_PATH)
    logger.info(f"Model successfully saved to: {config.MODEL_PATH}")
    
    return model, preprocessor, X_test, y_test


if __name__ == "__main__":
    train_model()
