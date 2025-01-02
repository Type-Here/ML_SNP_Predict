"""
    This module contains the functions to balance the data for the p53 dataset.
    4. This module should be the fourth module in the pipeline.
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

def balance_data(data: pd.DataFrame) -> tuple:
    """
        Balance the data with SMOTE. \n
        Split the data into features and target.
        Perform SMOTE on the data.     
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