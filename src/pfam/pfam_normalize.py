"""
    This module is responsible for normalizing the Pfam data.
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_pfam_data(dataset:pd.DataFrame, in_range:tuple = None, out_range:tuple = (0, 1)) -> pd.DataFrame:
    """
        Normalize the Pfam score data inside a Dataset.
        If in_range is None, MinMaxScaler will be used.
        If in_range is not None, the data will be normalized using the given range.

        Parameters:
            dataset (pd.DataFrame): dataset containing the Pfam data in the `Conservation` column.
            in_range (tuple): The range of the input data. Manually give range of possible input values to normalize from. 
                              Default is None: MinMaxScaler will be used.
            out_range (tuple): The range to normalize the data to. Default is (0, 1).
        Returns:
            pd.DataFrame: The dataset with normalized Pfam data.
    """
    if in_range is None:
         # Normalize the Pfam data using MinMaxScaler
        scaler = MinMaxScaler(feature_range=range)
        dataset["Conservation"] = scaler.fit_transform(dataset["Conservation"])
        return dataset
    
    else:
        # Manually reapply a MinMaxScaler to the Conservation column
        # to normalize it between 0 and 1.
        seq_min = in_range[0]
        seq_max = in_range[1]
    
        dataset['Conservation'] = dataset["Conservation"] \
            .apply(lambda x: ((x - seq_min) / (seq_max - seq_min)) * (out_range[1] - out_range[0]) + out_range[0])

        return dataset