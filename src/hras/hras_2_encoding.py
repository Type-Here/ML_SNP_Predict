"""
    This module contains the functions to encode the HRAS protein sequence into a numerical format (Encoding).
    2. This should be the second module to be executed for Hras.
"""
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re

def hras_data_encoding(dataset: pd.DataFrame, pfam = True, 
                       isPrediction = False, use_P53_V2 = True) -> pd.DataFrame:
    """
    Encode the HRAS data.
    1. Encode the `AA Ref` and `AA Mut` columns.
    2. Encode the `cDNA Ref` and `cDNA Mut` columns.
    3. Encode the `Domain` column.
    4. Encode the `Pathogenicity` column.

    Parameters:
        dataset (pd.DataFrame): The HRAS dataset to encode.
        pfam (bool): If the dataset will use Pfam for Domain evaluation. Default is True
        isPrediction (bool): If the data is for prediction. Default is False.
        use_P53_V2 (bool): If the V2 P53 Model will be used for Transfer Learning. Default is True.

    Returns:
        pd.DataFrame: The encoded HRAS data.
    """
    # Encoding #
    
    # Encode the `AA Ref` and `AA Mut` columns
    dataset = __encode_aa(dataset, use_P53_V2)
    
    # Encode the `cDNA Ref` and `cDNA Mut` columns
    dataset = __encode_cdna(dataset, use_P53_V2)
    
    # Encode the `Domain` column
    if pfam:
        # Encode the `Domain` column using Pfam
        #dataset = __encode_domain_pfam(dataset) #TODO: Implement Pfam Domain assignment
        pass
    else:
        dataset = __encode_domain_basic(dataset, use_P53_V2)
    
    if isPrediction:
        return dataset
    
    # Encode the `Pathogenicity` column
    dataset = __encode_pathogenicity(dataset)

    # Check non-numeric columns (if any). Prints the info
    __check_non_numeric(dataset)
    
    return dataset

# ------------------------- One-Hot Encoding ------------------------- #

def __one_hot_encode(data: pd.DataFrame, columns_to_encode: list, valid_categories: list) -> pd.DataFrame:
    """
        One-hot encode the selected columns in the dataset.
        Parameters:
            data (pd.DataFrame): The dataset to encode.
            columns_to_encode (list): The columns to encode.
            valid_categories (list): The valid categories for encoding.
        Returns:
            pd.DataFrame: The dataset with the selected columns one-hot encoded.
    """
    # Apply one-hot encoding to cDNA_Ref and cDNA_Mut
    encoder = OneHotEncoder(categories=[valid_categories] * len(columns_to_encode),
                            sparse_output=False, 
                            handle_unknown='ignore')

    # Fit-transform of selected columns
    encoded_matrix = encoder.fit_transform(data[columns_to_encode])

    # Verify the shape of the encoded matrix
    print("\nShape of one-hot encoded matrix:")
    print(encoded_matrix.shape)

    # Replace the original columns with the matrix 
    data_replaced = data.drop(columns=columns_to_encode)

    # Convert matrix to DataFrame
    pd_encoded_matrix = pd.DataFrame(
        encoded_matrix,  # The dense matrix (numpy.ndarray)
        index=data.index,
        columns=encoder.get_feature_names_out(columns_to_encode)
    )

    data_replaced = pd.concat([data_replaced, pd_encoded_matrix], axis=1)
    return data_replaced


# ------------------------- Ordinal Encoding ------------------------- #

def __ordinal_encoding(dataset, columns_to_encode, categories):
    """
        Encode the specified columns using ordinal encoding.
        The categories are mapped to integer values.

        Parameters:
            dataset (pd.DataFrame): The DataFrame containing the data to be encoded.
            columns_to_encode (list): The list of columns to be encoded.
            categories (list): The list of valid categories for each column to be encoded.
        Returns:
            pd.DataFrame: The DataFrame containing the encoded data.
    """
    cat_mapping = {v: i for i, v in enumerate(categories)}

    for col in columns_to_encode:
        dataset[col + "_Encoded"] = dataset[col].map(cat_mapping)
    
    # Drop original columns
    dataset = dataset.drop(columns=columns_to_encode)

    return dataset


# ------------------------- Column Encoding Functions ------------------------- #


def __encode_cdna(dataset: pd.DataFrame, use_P53_V2: bool) -> pd.DataFrame:
    """
    Encode the `cDNA Ref` and `cDNA Mut` columns.

    Parameters:
        dataset (pd.DataFrame): The HRAS dataset to encode.
        use_P53_V2 (bool): If the V2 P53 Model will be used for Transfer Learning.

    Returns:
        pd.DataFrame: The HRAS dataset with encoded `cDNA Ref` and `cDNA Mut` columns.
    """
    nucleotide_categories = ['A', 'T', 'C', 'G']
    # Combine columns to encode
    columns_to_encode = ['cDNA_Ref', 'cDNA_Mut']

    if use_P53_V2:
        return __ordinal_encoding(dataset, columns_to_encode, nucleotide_categories)

    return __one_hot_encode(dataset, columns_to_encode, nucleotide_categories)


# -- Start of AA Encoding -- #

def __encode_aa(dataset: pd.DataFrame, use_P53_V2:bool) -> pd.DataFrame:
    """
    Encode the `AA Ref` and `AA Mut` columns.

    Parameters:
        dataset (pd.DataFrame): The HRAS dataset to encode.
        use_P53_V2 (bool): If the V2 P53 Model will be used for Transfer Learning.

    Returns:
        pd.DataFrame: The HRAS dataset with encoded `AA Ref` and `AA Mut` columns.
    """
    aa_categories = ['G','A','V','L','I','T','S','M','C','P','F','Y','W','H','K','R','D','E','N','Q','0']

    # Combine columns to encode
    columns_to_encode = ['WT AA_1','Mutant AA_1']

    for v in columns_to_encode:
        dataset[v] = dataset[v].apply(__clean_amino_acids_introns)
    
    if use_P53_V2:
        return __ordinal_encoding(dataset, columns_to_encode, aa_categories)
    
    return __one_hot_encode(dataset, columns_to_encode, aa_categories)


def __clean_amino_acids_introns(variant):
    if variant == '*':
        return '0'
    if re.match(r'^\w$', variant):
        return variant
    else:
        return '0'
    
# -- End of AA Encoding -- #


def __encode_domain_basic(dataset: pd.DataFrame, use_P53_V2: bool) -> pd.DataFrame:
    """
        Encode the `Domain` column using basic domain assignment.
        Uses one-hot encoding for the `Domain` column.

        Parameters:
            data (pd.DataFrame): The dataset to encode.
            use_P53_V2 (bool): If the V2 P53 Model will be used for Transfer Learning.
        Returns:
            pd.DataFrame: The dataset with the `Domain` column encoded.    
    """
    # Encoding for remaining categorical columns
    categorical_columns = ['Domain']

    if use_P53_V2:
        return __ordinal_encoding(dataset, categorical_columns, dataset['Domain'].unique())

    return __one_hot_encode(dataset, categorical_columns, dataset['Domain'].unique())


def __check_non_numeric(data):
    """
        Check the non-numeric columns in the dataset.
        Parameters:
            data (pd.DataFrame): The dataset to check.
        Prints:
            The non-numeric columns in the dataset.
    """
    # Check the data types of the features
    print("\nFeature types in Dataset:")
    print(data.dtypes)

    # Identify non-numeric columns
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    print("\nNon-numeric columns in Dataset:")
    print(non_numeric_columns)



def __encode_pathogenicity(data: pd.DataFrame) -> pd.DataFrame:
    """
        Encode the `Pathogenicity` column.
        Maps:
            'Benign' -> 0
            'Pathogenic' -> 1
            'VUS' or 'not provided' -> 2
        Parameters:
            data (pd.DataFrame): The dataset to encode.
        Returns:
            pd.DataFrame: The dataset with the `Pathogenicity` column encoded.
    """
    # Map all unique values in 'Pathogenicity' to numeric classes
    data['Pathogenicity'] = data['Pathogenicity'].map({
        'Benign': 0,
        'Benign/Likely benign':0,
        'Likely benign' : 0,
        'Pathogenic': 1,
        'Pathogenic/Likely pathogenic' : 1,
        'Likely Pathogenic': 1,
        'Likely pathogenic' : 1,
        'VUS': 2,
        'Possibly pathogenic': 2,
        'Possibly Pathogenic': 2, # Adjust capitalization if needed
        'Uncertain significance': 2,
        'Conflicting classifications of pathogenicity': 2,
        'not provided': 2
    })
    return data


def __check_for_null_path_after_enc(data):
    """
        Check for NaN values in the 'Pathogenicity' column after encoding.
        Prints the rows with NaN values in the 'Pathogenicity' column.
        Parameters:
            data (pd.DataFrame): The dataset to check.
        Prints:
            The rows with NaN values in the 'Pathogenicity' column.
    """
    # Check for NaN values after mapping
    if data['Pathogenicity'].isnull().any():
        print("\nRows with NaN in 'Pathogenicity':")
        print(data[data['Pathogenicity'].isnull()])
        # Drop rows with NaN in 'Pathogenicity' (or handle as appropriate)
        data = data.dropna(subset=['Pathogenicity'])

    # Verify class distribution after cleaning
    print("\nUpdated class distribution in 'Pathogenicity':")
    print(data['Pathogenicity'].value_counts())