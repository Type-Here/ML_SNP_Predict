"""
    This module contains the functions for data processing of the HRAS dataset.
    1. This schould be the first module to be executed for Hras.
"""

import pandas as pd
import numpy as np
import re

def hras_data_cleaning(dataset: pd.DataFrame, isPfam = False) -> pd.DataFrame:
    """
    Clean the HRAS data.
    1. Filter rows to keep only SNPs.
    2. Remove unnecessary columns.
    3. Extract cDNA Ref and Mut.
    4. Filter out rows where the 'Name' column starts with "NC_".
    5. Extract `Position` from `Name` column.
    6. Extract AA Ref and Mut.
    7. Drop of `Name` column as it is not useful anymore.
    8. Add `Domain` column.
    9. Rename `Germline classification` column in `Pathogenicity` for consistency.

    Parameters:
        dataset (pd.DataFrame): The HRAS dataset to clean.
        isPfam (bool): If the dataset will use Pfam for Domain evaluation. Default is False.

    Returns:
        pd.DataFrame: The cleaned HRAS data.
    """
    # Data Cleaning #

    # Filter rows: only SNPs
    data = __filter_rows(dataset)

    # Remove unnecessary columns
    data = __drop_columns(data)

    # Feature Selection #

    # Extract cDNA Ref and Mut
    data = __ref_mut_cDNA_extraction(data)

    # Filter out rows where the 'Name' column starts with "NC_" 
    # NC: Chromosome RefSeq not useful for our analysis
    data = data[~data['Name'].str.startswith('NC_', na=False)]

    # Extract `Position` from `Name` column
    data = __extract_position(data)

    # Extract AA Ref and Mut
    data = __extract_aa_ref_mut(data)
    
    # Drop of `Name` column as it is not useful anymore
    data = data.drop(columns=['Name'])

    # Add `Domain` column
    if isPfam:
        #data = __assign_domain_pfam(data) #TODO: Implement Pfam Domain assignment
        pass
    else:
        data = __assign_domain_basic(data)

    # Rename `Germline classification` column in `Pathogenicity` for consistency
    data = data.rename(columns={'Germline classification': 'Pathogenicity'})

    return data


def __filter_rows(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter rows to keep only SNPs.
    """
    # Filter by 'Variant type' : 'single nucleotide variant'
    return data[data['Variant type'] == 'single nucleotide variant']


def __drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unnecessary columns from the HRAS data.
    """
    # Columns to Remove
    columns_to_drop = [
        'Protein change', 'Condition(s)', 'Accession', 'GRCh37Chromosome', 'GRCh37Location' ,'GRCh38Chromosome',
        'GRCh38Location', 'VariationID', 'AlleleID(s)', 'dbSNP ID', 'Canonical SPDI',
        'Variant type', 'Germline date last evaluated', 'Germline review status',
        'Somatic clinical impact', 'Somatic clinical impact date last evaluated',
        'Somatic clinical impact review status', 'Oncogenicity classification',
        'Oncogenicity date last evaluated', 'Oncogenicity review status', 'Gene(s)', 'Molecular consequence'
    ]

    # Drop Columns
    return data.drop(columns=columns_to_drop, errors='ignore')


def __ref_mut_cDNA_extraction(data: pd.DataFrame) -> pd.DataFrame:
    """
        Extract the reference and mutant cDNA nucleotides from the 'Name' column.
    """
    # Extract wild-type and mutant nucleotides
    data['cDNA_Ref'] = data['Name'].str.extract(r'[cg]\.[\d*+-]+([A-Z])>', expand=False)
    data['cDNA_Mut'] = data['Name'].str.extract(r'[cg]\.[\d*+-]+[A-Z]>([A-Z])', expand=False)

    return data

# -- Extract Position -- #

def __extract_position(data: pd.DataFrame) -> pd.DataFrame:
    """
        Extract the position from the 'Name' column.
    """
    # Apply the function to cDNA_variant
    data['cDNA_Position'] = data['Name'].apply(calculate_position)

    return data
    
# Function to calculate absolute position from cDNA_variant
def calculate_position(variant):
    try:
        # Remove nucleotide change (e.g., G>A, T>C)
        variant_cleaned = re.sub(r'[A-Z]>[A-Z].*', '', variant)
        variant_cleaned = re.sub(r'.*c.', '', variant_cleaned)

        # Handle 3' UTR positions like "c.*5325"
        if variant_cleaned.startswith('*'):
            variant_cleaned = variant_cleaned[1:]

        # Handle downstream intronic positions like "c.-29+1002" or "c.30+200"
        if '+' in variant_cleaned:
            parts = variant_cleaned.split('+')
            if len(parts) == 2:
                base, offset = parts
                return int(base) + int(offset)

        # Handle upstream intronic positions like "c.-28-1001"
        if '-' in variant_cleaned:
            parts = variant_cleaned.split('-')
            if len(parts) == 2:
                base, offset = parts
                if base == '':
                  base = '0'
                return int(base) - int(offset)
            if len(parts) == 3:
                empty, base, offset = parts
                return - int(base) - int(offset)

        # Handle simple numeric positions like "c.742"
        if re.match(r'\d+$', variant_cleaned):
            return int(re.search(r'(\d+)', variant_cleaned).group(1))

        # Return NaN for unhandled cases
        print(f"No error but -1: {variant}")
        return -1
    except Exception as e:
        print(f"Error processing variant {variant}: {e}")
        return np.nan


def __check_for_null_positions(data: pd.DataFrame) -> pd.DataFrame:
    """
    Check for null values in the HRAS data 'cDNA_Position' column.
    """
    # Check rows where cDNA_Position is -1 or NaN
    invalid_positions = data[data['cDNA_Position'].isna() | (data['cDNA_Position'] == -1)]

    # Count and display examples of problematic rows
    print(f"\nNumber of rows with invalid positions: {invalid_positions.shape[0]}")
    print("Sample of rows with invalid positions:")
    print(invalid_positions[['Name', 'cDNA_Position']].head(10))

# -- End Extract Position -- #


# -- Extract AA Ref and Mut -- #

def __extract_aa_ref_mut(data: pd.DataFrame) -> pd.DataFrame:
    """
        Extract the reference and mutant amino acids from the 'Protein change' column.
    """
    # Extract wild-type and mutant amino acids
    data[['WT_AA_1', 'Mutant_AA_1']] = data['Name'].apply(
                            extract_amino_acids).apply(pd.Series)
    return data

def extract_amino_acids(name):
    try:
        # Case 1: Intronic (no parentheses)
        if '(' not in name:
            return '0', '0'

        # Extract the amino acid change within parentheses
        aa_change = re.search(r'\((p\..*?)\)', name)
        if not aa_change:
            return '0', '0'

        aa_change = aa_change.group(1)

        # Case 2: No mutation (contains "=") (es. `p.Ser189=`)
        if '=' in aa_change:
            wt_aa = re.search(r'p\.([A-Za-z]+)\d+', aa_change)
            if wt_aa:
                wt_aa = wt_aa.group(1)
                return amino_acid_map.get(wt_aa, '0'), amino_acid_map.get(wt_aa, '0')

        # Case 3: Mutation (e.g., p.Ser189Ala)
        match = re.match(r'p\.([A-Za-z]+)\d+([A-Za-z]+)', aa_change)
        if match:
            wt_aa = amino_acid_map.get(match.group(1), '0')
            mutant_aa = amino_acid_map.get(match.group(2), '0')
            return wt_aa, mutant_aa

        # Default return if nothing matches
        return '0', '0'
    except Exception as e:
        print(f"Error processing: {name}, {e}")
        return np.nan, np.nan
    
# -- End Extract AA Ref and Mut -- #



# -- Assign Domain -- #

def __assign_domain_basic(data: pd.DataFrame) -> pd.DataFrame:
    """
        Assign the functional domain based on the cDNA position and other conditions.
        This is a basic domain assignment based on the cDNA position.
    """
    # Apply the function to assign domains
    data['Domain'] = data.apply(__assign_domain_by_position, axis=1)


def __assign_domain_by_position(row):
    """
        Assign the functional domain based on the cDNA position.
    """
    position = row['cDNA_Position']
    wt_aa = row['WT_AA_1']

    # Check if WT AA_1 is '0', indicating it is an intron
    if wt_aa == '0':
        return 'Intron'

    # Determine the domain based on the position
    if position < 0:
        return 'UTR'
    elif 1 <= position <= 162:
        return 'G1-G3'
    elif 163 <= position <= 324:
        return 'G4-G5'
    elif 325 <= position <= 432:
        return 'Switch I'
    elif 433 <= position <= 540:
        return 'Switch II'
    elif 541 <= position <= 567:
        return 'C-terminal'
    else:
        return 'Other'

# -- End Assign Domain -- #





# Amino Acid Map #
"""
    Amino acid three-letter to one-letter code mapping.
"""
amino_acid_map = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Glu': 'E', 'Gln': 'Q', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    # Stop codon
    'Ter': '*'
}