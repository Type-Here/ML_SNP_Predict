"""
    This module contains the functions to normalize the HRAS dataset
    3. This should be the third module to be executed for Hras.
"""
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math

from src.pfam.pfam_normalize import normalize_pfam_data

def hras_scaling(dataset:pd.DataFrame, pfam = True) -> pd.DataFrame:
    """
        Apply Data Scaling to HRAS Dataset.
        1. Normalize `cDNA_Position` with MinMaxScaler
        Params:
            dataset(pd.DataFrame): hras dataset to normalize
            pfam(bool): Whether to normalize the Pfam data. Default is True
        Returns:
            dataset(pd.DataFrame): hras dataset normalized
    """
    if pfam:
        # Normalize the Pfam data
        dataset = __pfam_conservation_norm(dataset)

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

    return data



def __pfam_conservation_norm(data:pd.DataFrame) -> pd.DataFrame:
    """
        Normalize the Pfam conservation data to the range [0, 1].

        Operations:
            - Operate -ln(x) on the conservation data.
            - Apply MinMaxScaler: a maximum value of 100 will be used. 
            - Output range [0, 1].

        Parameters:
            data (pd.DataFrame): The DataFrame containing the Pfam conservation data.
        Returns:
            pd.DataFrame: The DataFrame with the normalized Pfam conservation data.\n

        Explaination
        ------------
        - This function performs normalization of E-values by applying 
        the negative natural logarithm transformation (-ln) to shift the values to a linear scale.\n 
        - It then applies Min-Max scaling, where the maximum theoretical value for normalization is set to 100.\n
        - The normalized output ranges from 0 to 1, ensuring consistency and comparability 
        across datasets while preserving the sensitivity of significant values.\n
        - This approach is suitable for machine learning models requiring scaled input.
        -  By using a maximum of 100, the function better differentiates values 
        in the range of 10^-20 to 10^-80, preserving the sensitivity of significant values 
        and enhancing the granularity of distinctions in this critical range. 
    """
    # Apply -ln(x) to the conservation data
    data['Conservation'] = data['Conservation'].apply(lambda x: -1 * math.log(x) if x > 0 else x)

    return normalize_pfam_data(data, in_range=(0, 100), out_range=(0, 1))