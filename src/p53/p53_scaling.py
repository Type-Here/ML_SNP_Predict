"""
    This module contains the functions to scale and normalize data in the p53 dataset.
    3. This module should be the third module in the pipeline.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def dna_position_norm(data:pd.DataFrame) -> pd.DataFrame:
    """
        Normalize the cDNA_Position data.\n
        Normalize the DNA position data to the range [0, 1].
    """
    

    # List of numerical columns to normalize
    numerical_columns = ['cDNA_Position']

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    # Verify the normalized dataset
    print("\nDataset after normalizing numerical columns:")
    print(data[numerical_columns].head())

    return data

