"""
    This module contains the functions to normalize the HRAS dataset
    3. This should be the third module to be executed for Hras.
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def hras_scaling(dataset:pd.DataFrame) -> pd.DataFrame:
    """
        Apply Data Scaling to HRAS Dataset.
        1. Normalize `cDNA_Position` with MinMaxScaler
        Params:
            dataset(pd.DataFrame): hras dataset to normalize
        Returns:
            dataset(pd.DataFrame): hras dataset normalized
    """
    return __norm_position(dataset)



def __norm_position(data, min = 0, max = 1):
    """
        Normalize `cDNA_Position` using MinMaxScaler
        Params:
            data (pd.DataFrame): dataset with cDNA_Position column
            min (int): minimum value to scale to. Default is 0
            max (int): maximum value to scale to. Default is 1
    """
    # List of numerical columns to normalize
    numerical_columns = ['cDNA_Position']

    # Apply MinMaxScaler
    scaler = MinMaxScaler(feature_range=(min, max))
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])