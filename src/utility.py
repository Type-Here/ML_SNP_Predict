"""
    Utility functions for the project
"""

import pandas as pd


def check_column_types(dataset: pd.DataFrame) -> int:
    """
        Check the types of the columns in the DataFrame.
        Print the data types of the features and identify non-numeric columns.

        Args:
            dataset (pd.DataFrame): The DataFrame to check.
        Returns:
            int: The number of non-numeric columns in the DataFrame.
    """
    
    # Check the data types of the features
    print("\nFeature types in Dataset:")
    print(dataset.dtypes)

    # Identify non-numeric columns
    non_numeric_columns = dataset.select_dtypes(include=['object']).columns
    print("\nNon-numeric columns in Dataset:")
    print(non_numeric_columns)

    return non_numeric_columns.size



def print_columns_with_nan(dataset:pd.DataFrame):
    """
        Print columns with NaN values in the DataFrame.

        Args:
            dataset (pd.DataFrame): The DataFrame to check.
    """
    # Print the columns
    print("\nColumns with at least one NaN value:")
    print(dataset.columns[dataset.isnull().any()].tolist())


def print_nan_rows(dataset:pd.DataFrame, columns: list = None):
    """
        Print rows with NaN values in the DataFrame.

        Args:
            dataset (pd.DataFrame): The DataFrame to check.
            columns (list): The list of columns to check for NaN values. Default is None: check all columns.
    """
    
    if columns:
        print("\nRows with NaN values in specific columns:")
        print(dataset[dataset[columns].isnull().any(axis=1)])
    else:
        # Print rows with NaN values
        print("\nRows with NaN values:")
        print(dataset[dataset.isnull().any(axis=1)])


def print_unique_values(dataset:pd.DataFrame, columns: list = None):
    """
        Print unique values in the specified columns of the DataFrame.

        Args:
            dataset (pd.DataFrame): The DataFrame to check.
            columns (list): The list of columns to check for unique values. Default is None: check all columns.
    """
    if columns is None:
        columns = dataset.columns

    for col in columns:
        print(f"\nUnique values in '{col}':")
        print(dataset[col].unique())


def print_sum_of_distinct_values(dataset:pd.DataFrame, columns: list = None):
    """
        Print the sum of distinct values in the specified columns of the DataFrame.

        Args:
            dataset (pd.DataFrame): The DataFrame to check.
            columns (list): The list of columns to check for sum of values. Default is None: check all columns.
    """
    if columns is None:
        columns = dataset.columns

    for col in columns:
        print(f"\nSum of distinct values in '{col}':")
        print(dataset[col].value_counts().sum())