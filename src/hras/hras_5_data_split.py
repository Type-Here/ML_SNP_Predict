"""
    This module contains the functions to create the training and test sets for the hras dataset.
    5. This module should be the fifth module in the pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_sets(X_data, y_labels) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits the input data into training and test sets.
    Parameters:
        X_data (pd.DataFrame): The input features.
        y_labels (pd.Series): The target labels.
    Returns:
        tuple: A tuple containing the training features (X_train), test features (X_test),
               training labels (y_train), and test labels (y_test).
    """
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

    print("\nTraining and test set sizes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test