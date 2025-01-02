"""
    This module contains the functions to create the training and test sets for the p53 dataset.
"""


import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test_sets(X_data, y_labels) -> tuple:
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels, test_size=0.2, random_state=42)

    print("\nTraining and test set sizes:")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    return X_train, X_test, y_train, y_test