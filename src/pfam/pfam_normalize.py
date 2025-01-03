"""
    This module is responsible for normalizing the Pfam data.
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_pfam_data(dataset:pd.DataFrame, range:tuple = (0, 1)) -> pd.DataFrame:
    """
        Normalize the Pfam score data inside a Dataset.
        Parameters:
            dataset (pd.DataFrame): dataset containing the Pfam data in the `Conservation` column.
        Returns:
            pd.DataFrame: The dataset with normalized Pfam data.
    """
    # Normalize the Pfam data
    scaler = MinMaxScaler(feature_range=range)
    dataset["Conservation"] = scaler.fit_transform(dataset["Conservation"])

    return dataset