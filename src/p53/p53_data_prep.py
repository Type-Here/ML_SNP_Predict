import pandas as pd
import numpy as np
import re


def p53_cleaning(dataset: pd.DataFrame, pfam: bool = False) -> pd.DataFrame:
    if dataset.empty:
        return None
    
    data = _filter_by_snp(dataset)
    data = _drop_columns(data)
    data = _nan_handling(data, pfam)
    data = _data_treatment(data)
    data = _feature_derivation(data)

    return data


# -----------------  Data Cleaning Functions ----------------- #

def _filter_by_snp(data):
    """
        Filters the data to keep only SNP mutations.
    """
    # Filter only SNP mutations
    snp_data = data[data['Variant_Type'] == 'SNP']
    print(f"Mutazioni SNP trovate: {snp_data.shape[0]}")

    return snp_data


def _drop_columns(data):
    """
        Drops unnecessary columns from the dataset.
    """
    # Columns to drop based on redundancy or irrelevance
    columns_to_drop = [
        #IDs
        'UMD_ID', 'COSMIC_ID', 'SNP_ID',
        
        #Protein data (redundant)
        *[col for col in data.columns if col.startswith('Transcript')],
        *[col for col in data.columns if col.startswith('Protein')],
        
        #Genomic data
        'HG19_Variant', 'HG18_Variant', 'HG38_Variant', 'NG_017013.2_Variant',  # Redundant genomic variant info
        'Start_cDNA', 'End_cDNA', 'Exon:intron_Start', 'Exon:intron_End',  # Derived info
        'Genome_base_coding', 'Mutant_Allele', 'Base_Change_Size', 'Ins_Size', 'Del_Size',  # Technical details or not used
        'PTM', 'Variant_Comment',  # Specific but not useful
        *[col for col in data.columns if col.startswith('Comment')],  # All comment-related columns
        'Sift_Prediction', 'Polyphen-2_HumVar', 'Provean_prediction', 'Condel',  # Computational predictions
        'Leukemia_Lymhoma_Freq', 'Solid_Tumor_Freq', 'Cell_line_Freq',  # Redundant frequency columns

        'HG19_Start', 'HG19_End', 'HG18_Start', 'HG18_End',  # Genomic positions
        'WAF1_Act', 'MDM2_Act', 'BAX_Act', '_14_3_3_s_Act', 'AIP_Act', 'GADD45_Act', 'NOXA_Act', 'p53R2_Act',  # Residual activities
        'WAF1_percent', 'MDM2_percent', 'BAX_percent', '14_3_3_s_percent', 'AIP_percent', 'GADD45_percent', 'NOXA_percent', 'p53R2_percent',  # Percent activities
        'Sift_Score', 'Polyphen-2_HumDiv', 'Mutassessor_prediction', 'Mutassessor_score', 'Provean_Score', 'Condel_Score',  # Computational predictions
        'MutPred_Splice_General_Score', 'MutPred_Splice_Prediction_Label', 'MutPred_Splice_Confident_Hypotheses',  # Splice predictions
        'Final comment',  # Textual comment
        'Records_Number',

        'CpG', 'Substitution type','Tumor_Freq','Somatic_Freq 2','Germline_Freq 2', # Little info or non used
        
        # Could be useful but not for this analysis
        'WT AA_3',
        'Mutant AA_3', 'Mutational_event', 'Variant_Classification',
        'Variant_Type', 'Mutation_Type'
        ]

    # Drop unnecessary columns
    snp_data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

    # Display remaining columns
    print("\nRemaining columns after dropping unnecessary ones:")
    print(snp_data_cleaned.columns)

    return snp_data_cleaned

# -----------------  End of Data Cleaning Functions ----------------- #


# -----------------  Nan Data Handling Functions ----------------- #

def _nan_handling(data, pfam):
    """
        Handles NaN values in the dataset.
    """
    # Drop rows with missing Pathogenicity: no label to predict (only 3 rows)
    snp_data_cleaned = data.dropna(subset=['Pathogenicity'])

    # Check for missing values
    missing_values = snp_data_cleaned.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values[missing_values > 0])

    # Drop 'Tandem_Class' and 'Structure' columns
    columns_to_drop = ['Tandem_Class', 'Structure']
    snp_data_cleaned = snp_data_cleaned.drop(columns=columns_to_drop)
    print("\nColumns 'Tandem_Class' and 'Structure' dropped.")


    if pfam:
        print("\nHandling NaN values in Pfam columns...") # TODO: Implement Pfam handling
    else:
        snp_data_cleaned = _domain_imputation(snp_data_cleaned)


    return snp_data_cleaned



def _domain_imputation(data):
    """
        Imputes missing 'Domain' values based on the closest known 'Codon'. \n
        Params:
            data (pd.DataFrame): The DataFrame containing the data.
        Returns:
            pd.DataFrame: The DataFrame with imputed 'Domain' values.    
    """
    # Step 1: Split dataset into rows with and without 'Domain'
    domain_missing = data[data['Domain'].isnull()]
    domain_known = data[~data['Domain'].isnull()]

    # Apply the function to impute missing 'Domain' values
    imputed_domains = []
    for index, row in domain_missing.iterrows():
        closest_domain = _find_closest_or_default_domain(row['Codon'], domain_known)
        imputed_domains.append(closest_domain)

    # Assign the imputed domains back to the missing rows
    data.loc[domain_missing.index, 'Domain'] = imputed_domains

    # Verify the result
    print("\nUpdated 'Domain' distribution:")
    print(data['Domain'].value_counts())

    return data


# Function to find the closest domain or return 'Intron' for missing codons
def _find_closest_or_default_domain(codon, known_data):
    """
        Find the closest domain to the given codon based on the known data.
        If the codon is not found, return 'Intron'. \n
        Params:
            codon (int): The codon to find the closest domain for.
            known_data (pd.DataFrame): The DataFrame containing the known codons and domains.
        Returns:
            str: The closest domain to the given codon.   
    """
    try:
        # Convert Codon to integer for numeric comparison
        codon = int(codon)
        # Calculate distances to known codons and find the closest
        known_data = known_data.copy()  # Avoid SettingWithCopyWarning
        known_data['Codon_Distance'] = abs(known_data['Codon'].astype(int) - codon)
        closest_row = known_data.loc[known_data['Codon_Distance'].idxmin()]
        return closest_row['Domain']
    except (ValueError, TypeError):
        # If codon is not a valid number, assume it's an intron
        return 'Intron'
    


## ------------------ End of nan_handling ------------------ ##


# -----------------  Data Treatment Functions ----------------- #

def _data_treatment(data):
    """
        Treats the data for the p53 protein.
        Codon: Converts 'Codon' column to numeric values and replaces non-numeric values with -1.
        WT_Codon and Mutant_Codon: Standardizes values by replacing non-standard strings with 'III'.
        WT AA_1 and Mutant AA_1: Cleans amino acids and introns.
    """

    # -- Codon column -- #

    # Clean 'Codon' column by removing everything after and including the dash ('-')
    data['Codon'] = data['Codon'].str.replace(r'-.*', '', regex=True)

    ##-------------------------------#
    # Replace values in 'Codon' containing 'ter' or 'Ter' with 0
    data.loc[data['Codon'].str.contains(r'ter', case=False, na=False), 'Codon'] = 0

    # Replace non-numeric values in 'Codon' with -1
    data['Codon'] = data['Codon'].apply(lambda x: -1 if not str(x).isdigit() else int(x))

    # Verify the result
    print("\nUpdated 'Codon' column (non-numeric replaced with -1):")
    print(data['Codon'].unique())


    # -- WT_Codon and Mutant_Codon columns -- #

    # Standardize WT_Codon and Mutant_Codon by replacing non-standard strings with 'III'
    # Some values contain non-standard characters or contain info like 'Splice': replace them with 'III'
    columns_to_update = ['WT_Codon', 'Mutant_Codon']

    for col in columns_to_update:
        data[col] = data[col].apply(
            lambda x: 'III' if not isinstance(x, str) or not x.isalpha() or len(x) != 3 else x.upper()
        )

    # Verify the result
    print("\nUpdated columns (non-standard replaced with 'III'):")


    # -- WT AA_1 and Mutant AA_1 columns -- #
    # -- Clean amino acids and introns -- #

    columns_to_check = ['WT AA_1','Mutant AA_1']
    for v in columns_to_check:
        data[v] = data[v].apply(_clean_amino_acids_introns)

    return data


def _clean_amino_acids_introns(variant):
  if variant == '*':
     return '0'
  if re.match(r'^\w$', variant):
    return variant
  else:
    return '0'


# -----------------  End of Data Treatment Functions ----------------- #

# -----------------  Feature Derivation Functions ----------------- #

def _feature_derivation(data):
    """
        Derives new features from existing columns.
        WT_Codon: Splits into three separate columns for each nucleotide.
        Mutant_Codon: Splits into three separate columns for each nucleotide.
        cDNA_Position: Extracts the position from cDNA_variant.
        cDNA_Ref and cDNA_Mut: Extracts the reference and mutant nucleotides from cDNA_variant.
    """
    print("\nDeriving new features...")
    
    data = _codons_split(data) # Split Codon into three separate columns for each nucleotide
    print("\nWT_Codon and Mutatn_Codon split into three separate columns each.")

    data = _cDNA_Variant_extraction(data) # Derive cDNA_Position, cDNA_Ref, and cDNA_Mut from cDNA_variant
    print("\ncDNA_Position, cDNA_Ref, and cDNA_Mut derived from cDNA_variant.")

    return data


def _codons_split(data):
    """
        Splits the 'WT_Codon' column into three separate columns for each nucleotide.
        Splits the 'Mutant_Codon' column into three separate columns for each nucleotide.
    """
     # Separate WT_Codon into its three nucleotides
    data['WT_Codon_First'] = data['WT_Codon'].str[0]
    data['WT_Codon_Second'] = data['WT_Codon'].str[1]
    data['WT_Codon_Third'] = data['WT_Codon'].str[2]

    # Separate Mutant_Codon into its three nucleotides
    data['Mutant_Codon_First'] = data['Mutant_Codon'].str[0]
    data['Mutant_Codon_Second'] = data['Mutant_Codon'].str[1]
    data['Mutant_Codon_Third'] = data['Mutant_Codon'].str[2]

    return data


def _cDNA_Variant_extraction(data):
    # Derive cDNA_Position from cDNA_variant
    data['cDNA_Position'] = data['cDNA_variant'].apply(_calculate_position)

    # Derive cDNA_Ref and cDNA_Mut from cDNA_variant
    data = _ref_and_mut_from_cDNA_Variant(data)

    # Remove cDNA_variant column
    data = data.drop(columns=['cDNA_variant'])

    return data


# Function to calculate absolute position from cDNA_variant
def _calculate_position(variant):
    try:
        # Remove nucleotide change (e.g., G>A, T>C)
        variant_cleaned = re.sub(r'[A-Z]>[A-Z]', '', variant)
        variant_cleaned = re.sub(r'c.', '', variant_cleaned)

        # Handle 3' UTR positions like "c.*5325"
        if variant_cleaned.startswith('*'):
            return int(variant_cleaned[1:])

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
    

def _ref_and_mut_from_cDNA_Variant(data):
    # Extract wild-type and mutant nucleotides
    data['cDNA_Ref'] = data['cDNA_variant'].str.extract(r'c\.[\d*+-]+([A-Z])>', expand=False)
    data['cDNA_Mut'] = data['cDNA_variant'].str.extract(r'c\.[\d*+-]+[A-Z]>([A-Z])', expand=False)

    return data


# -----------------  End of Feature Derivation Functions ----------------- #

