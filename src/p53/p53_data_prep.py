import pandas as pd

def p53_cleaning(dataset, pfam = False):
    if not dataset:
        return None
    
    data = filter_by_snp(dataset)
    data = drop_columns(data)
    data = nan_handling(data, pfam)


    return dataset




def filter_by_snp(data):
    """
        Filters the data to keep only SNP mutations.
    """
    # Filter only SNP mutations
    snv_data = data[data['Variant_Type'] == 'SNP']
    print(f"Mutazioni SNP trovate: {snv_data.shape[0]}")

    return snv_data


def drop_columns(data):
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

        'CpG', 'Substitution type','Tumor_Freq','Somatic_Freq 2','Germline_Freq 2' # Little info or non used
        ]

    # Drop unnecessary columns
    snp_data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')

    # Display remaining columns
    print("\nRemaining columns after dropping unnecessary ones:")
    print(snp_data_cleaned.columns)

    return snp_data_cleaned


def nan_handling(data, pfam):
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
        snp_data_cleaned = domain_imputation(snp_data_cleaned)


    return snp_data_cleaned


## -----------------  Used by nan_handling ----------------- ##

def domain_imputation(data):
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
        closest_domain = find_closest_or_default_domain(row['Codon'], domain_known)
        imputed_domains.append(closest_domain)

    # Assign the imputed domains back to the missing rows
    data.loc[domain_missing.index, 'Domain'] = imputed_domains

    # Verify the result
    print("\nUpdated 'Domain' distribution:")
    print(data['Domain'].value_counts())


# Function to find the closest domain or return 'Intron' for missing codons
def find_closest_or_default_domain(codon, known_data):
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


    