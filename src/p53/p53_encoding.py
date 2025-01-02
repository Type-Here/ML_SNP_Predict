import pandas as pd
from sklearn.preprocessing import OneHotEncoder
    

def p53_encoding(dataset:pd.DataFrame, pfam:bool = False) -> pd.DataFrame:
    """
        Encode the p53 data.
        Operations: \n
        One-hot Encoding of: \n
        - Encode the nucleotide data for the wild-type and mutant splitted codons.
        - Encode the cDNA reference and mutant data.
        - Encode the amino acid data of the wild-type and mutant columns.
        - Encode the p53 domain data. Used for the p53 dataset if the Pfam domain data is not available or not used.
        - Encode the pathogenicity data.
    
        Args:
            dataset (pd.DataFrame): The DataFrame containing the data to be encoded.
        
        Returns:
            pd.DataFrame: The DataFrame containing the encoded data.
    """
    
    # Encode the data

    data_encoded = _wt_mut_nucleotide_encoding(dataset)
    print("Nucleotide encoding done")

    data_encoded = _cDNA_ref_mut_encoding(data_encoded)
    print("cDNA encoding done")

    data_encoded = _amino_acid_encoding(data_encoded)
    print("Amino acid encoding done")

    if pfam:
        #data_encoded = pfam_domain_encoding(data_encoded) # TODO: Implement this function
        pass
    else:
        data_encoded = _p53_domain_encoding(data_encoded)
    print("p53 domain encoding done")

    data_encoded = _pathogenicity_encoding(data_encoded)
    print("Pathogenicity encoding done")

    return data_encoded



def _wt_mut_nucleotide_encoding(dataset):
    """
        Encode the nucleotide data for the wild-type and mutant splitted codons.
    """   
    # Define nucleotide values
    nucleotide_categories = ['A', 'T', 'C', 'G', 'I'] # 'I' is for Intronic Sequences modified to output III in some rows

     # Columns to encode
    columns_to_encode = ['WT_Codon_First', 'WT_Codon_Second', 'WT_Codon_Third',
                        'Mutant_Codon_First', 'Mutant_Codon_Second', 'Mutant_Codon_Third']

    # One-hot encoding
    data = __one_hot_encoding(dataset, columns_to_encode, nucleotide_categories)

    # Drop now unnecessary columns
    columns_to_drop = [
        'Codon', 'WT_Codon', 'Mutant_Codon'
    ]

    data_cleaned = data.drop(columns=columns_to_drop)

    return data_cleaned




def _cDNA_ref_mut_encoding(dataset):
    """
        Encode the cDNA reference and mutant data.
    """
    nucleotide_categories = ['A', 'T', 'C', 'G']
    # Columns to encode
    columns_to_encode = ['cDNA_Ref', 'cDNA_Mut']

    # One-hot encoding
    data_encoded = __one_hot_encoding(dataset, columns_to_encode, nucleotide_categories)

    return data_encoded


def _amino_acid_encoding(dataset):
    """
        Encode the amino acid data.
    """
    aa_categories = ['G','A','V','L','I','T','S','M','C','P','F','Y','W','H','K','R','D','E','N','Q','0']

    # Columns to encode
    columns_to_encode = ['WT AA_1','Mutant AA_1']

    # One-hot encoding
    data_encoded = __one_hot_encoding(dataset, columns_to_encode, aa_categories)

    return data_encoded



def _p53_domain_encoding(dataset):
    """
        Encode the p53 domain data. Used for the p53 dataset if the Pfam domain data is not available or not used.
    """

    # Encoding for remaining categorical columns

    categorical_columns = ['Domain']
    data_cleaned = pd.get_dummies(dataset, columns=categorical_columns)

    # Verify the dataset after cleaning and encoding
    # print("\nDataset X after cleaning and encoding:")
    # print(snv_data_cleaned.head())

    return data_cleaned



# ------------------------------ One-Hot Encoding ------------------------------ #

def __one_hot_encoding(dataset, columns_to_encode, categories):
    """
        Perform one-hot encoding on the specified columns.
    
        Args:
            dataset (pd.DataFrame): The DataFrame containing the data to be encoded.
            columns_to_encode (list): The list of columns to be encoded.
            categories (list): The list of valid categories for each column to be encoded.
        
        Returns:
            pd.DataFrame: The DataFrame containing the encoded data.
    """
    
    # Specificy valid categories for each column
    encoder = OneHotEncoder(categories=[categories] * len(columns_to_encode),
                            sparse_output=False,  # Sparse matrix output (False for dense)
                            handle_unknown='ignore')  # Ignore unknown values

    # Fit-transform of selected columns
    encoded_matrix = encoder.fit_transform(dataset[columns_to_encode])

    # Verify the shape of the encoded matrix
    # print("\nShape of one-hot encoded matrix:")
    # print(encoded_matrix.shape)

    # Replace original columns with encoded columns
    data_cleaned = dataset.drop(columns=columns_to_encode)
    # data_cleaned = pd.concat([dataset, pd.DataFrame.sparse.from_spmatrix(encoded_sparse_matrix)], axis=1) # If sparse_output=True
    
    # Convert dense matrix to DataFrame
    pd_encoded_matrix = pd.DataFrame(
        encoded_matrix,  # The dense matrix (numpy.ndarray)
        index=dataset.index,
        columns=encoder.get_feature_names_out(columns_to_encode)
    )
    # Drop original columns and concatenate encoded columns
    data_cleaned = dataset.drop(columns=columns_to_encode)
    data_cleaned = pd.concat([data_cleaned, pd_encoded_matrix], axis=1)

    # print("\nDataset after encoding:")
    # print(data_cleaned.head())

    return data_cleaned



# ------------------------------ Pathogenicity Encoding ------------------------------ #

def _pathogenicity_encoding(dataset):
    """
        Encode the 'Pathogenicity' column in the dataset.
        0: Benign 
        1: Pathogenic (Pathogenic, Likely Pathogenic)
        2: VUS (Variant of Uncertain Significance)
    """
    # Map all unique values in 'Pathogenicity' to numeric classes
    dataset['Pathogenicity'] = dataset['Pathogenicity'].map({
        'Benign': 0,
        'Pathogenic': 1,
        'Likely Pathogenic': 1,
        'VUS': 2,
        'Possibly pathogenic': 2,
        'Possibly Pathogenic': 2  # Adjust capitalization if needed
    })

    # Check for NaN values after mapping
    if dataset['Pathogenicity'].isnull().any():
        print("\nRows with NaN in 'Pathogenicity':")
        print(dataset[dataset['Pathogenicity'].isnull()])
        # Drop rows with NaN in 'Pathogenicity' (or handle as appropriate)
        dataset = dataset.dropna(subset=['Pathogenicity'])
        print("\nNaN rows dropped.")

    # Verify class distribution after cleaning
    #print("\nUpdated class distribution in 'Pathogenicity':")
    #print(snv_data_cleaned['Pathogenicity'].value_counts())

    return dataset


