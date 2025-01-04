"""
    This module contains the functions to normalize the HRAS dataset
    4. This should be the third module to be executed for Hras.
"""
import pandas as pd
from imblearn.over_sampling import SMOTE

def hras_data_balancing(dataset:pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
        Balance the data with SMOTE. \n
        Split the data into features and target.
        Perform SMOTE on the data.     

        Args:
            data (pd.DataFrame): The DataFrame containing the data to be balanced.
        Returns:
            tuple: The resampled data and the resampled target.


    """
    return __smote_resampling(dataset)



def __smote_resampling(data):
    """
        Perform SMOTE Resampling.
        Returns:
            tuple: The resampled data and the resampled target.
    """
    # Proceed with SMOTE
    X = data.drop(columns=['Pathogenicity'])
    y = data['Pathogenicity']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Verify class distribution after SMOTE
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_resampled).value_counts())
    return X_resampled, y_resampled