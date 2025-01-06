"""
    This module contains the functions to scale and normalize data in the p53 dataset.
    3. This module should be the third module in the pipeline.
"""

import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def p53_scaling(data:pd.DataFrame, pfam = False) -> pd.DataFrame:
    """
        Scale and normalize the p53 data.
        Operations:
        - Scale the cDNA_Position data.
        - Normalize the Pfam conservation data if pfam is True. \n
          MinMaxScaler is used with a maximum value of 100. \n
          See the __pfam_conservation_norm function for more details.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the p53 data.
            pfam (bool): Whether to normalize the Pfam data. Default is False.
        Returns:
            pd.DataFrame: The DataFrame with the scaled and normalized data.
    """

    # Scale the data
    data = __dna_position_norm(data)

    if pfam:
        # Normalize the Pfam data
        data = __pfam_conservation_norm(data)

    # Inform the user
    print("\nNormalized numerical columns.")

    return data

def __dna_position_norm(data:pd.DataFrame) -> pd.DataFrame:
    """
        Normalize the cDNA_Position data.\n
        Normalize the DNA position data to the range [0, 1].
    """
    

    # List of numerical columns to normalize
    numerical_columns = ['cDNA_Position']

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
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

    # Apply MinMax Scaling
    max_value = 100
    min_value = 0

    data['Conservation'] = data['Conservation'].apply(lambda x: (x - min_value) / (max_value - min_value)) # * (1 - 0) + 0)
    
    return data
