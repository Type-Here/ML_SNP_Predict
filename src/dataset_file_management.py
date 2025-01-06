import os
import json
import pandas as pd
import requests
import zipfile
from src.config import DATA_PATH, PFAM_PATH

def load_data(protein='p53'):
    """
        Load the data for the specified protein.
    
        Args:
            protein (str): The name of the protein for which the data is to be loaded. Default is 'p53'.
    
        Returns:
            pd.DataFrame: The DataFrame containing the data for the specified protein.
    """
    
    match protein:
        case 'p53': 
            return __ensure_p53_data()
        case 'hras':
            return __ensure_hras_data()
        case _: 
            ValueError(f"Protein {protein} not supported.")




# Internal function to ensure p53 data is available
def __ensure_p53_data():
    """
        Ensures that the p53 data file is available and ready for use.
    
        The function checks if the p53.csv file is present in the 'processed' directory.
        If the file is not present, it checks if it is available in the 'raw' directory.
        If the file is not present in the 'raw' directory either, it downloads it from a specified URL.
        Once downloaded, the file is saved in the 'raw' directory.
        Subsequently, the file is loaded and saved in the 'processed' directory.
    
        Returns:
            pd.DataFrame: The DataFrame containing the data from the p53.csv file.
    
        Raises:
            Exception: If the file download fails.
    """
    # File paths
    processed_path = os.path.join(DATA_PATH, 'processed', 'p53.csv')
    raw_zip_path = os.path.join(DATA_PATH, 'raw', 'UMD_variants_EU.tsv.zip')
    raw_tsv_path = os.path.join(DATA_PATH, 'raw', 'UMD_variants_EU.tsv')
    download_url = "https://p53.fr/images/Database/UMD_variants_EU.tsv.zip" 

    # Check if file exists in processed
    if os.path.exists(processed_path):
        print(f"File trovato in: {processed_path}")
        return pd.read_csv(processed_path)
    
    print(f"File non trovato in: {processed_path}")
    
    # Check if file exists in raw
    if not os.path.exists(raw_zip_path):
        print(f"Scaricamento del file da: {download_url}")

        response = requests.get(download_url)
        
        if response.status_code == 200:
            os.makedirs(os.path.join(DATA_PATH, 'raw'), exist_ok=True)
            with open(raw_zip_path, 'wb') as f:
                f.write(response.content)
            print(f"File ZIP scaricato e salvato in: {raw_zip_path}")
        else:
            raise Exception(f"Errore nel download. Status code: {response.status_code}")

    # Extract ZIP file 
    if not os.path.exists(raw_tsv_path):
        print(f"Estrazione del file ZIP: {raw_zip_path}")

        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(DATA_PATH, 'raw'))

        print(f"File TSV estratto in: {raw_tsv_path}")

    # Load file
    print(f"Caricamento del file TSV: {raw_tsv_path}")
    df = pd.read_csv(raw_tsv_path, sep='\t', encoding='latin-1')  # File TSV with \t separator and latin-1 encoding


    # Save file in processed
    #os.makedirs(os.path.join(DATA_PATH, 'processed'), exist_ok=True)
    #df.to_csv(processed_path, index=False)
    #print(f"File salvato in: {processed_path}")
    
    return df

# ------------------------------ HRAS Data ------------------------------ #

def __ensure_hras_data():
    """
        Ensures that the HRAS data file is available and ready for use.
        Loads the HRAS data from the 'raw' directory.

        Note: The HRAS data is already available in the 'raw' directory and does not need to be downloaded.
              This because source (LOVD) does not provide a simple API to download the data.
        Returns:
            pd.DataFrame: The DataFrame containing the HRAS data.
        Raises:
            
    """
    raw_path = os.path.join(DATA_PATH, 'raw', 'hras.csv')

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File HRAS non trovato in: {raw_path}")

    return pd.read_csv(raw_path, delimiter='\t', encoding='latin-1')



# ------------------------------ Save Dataset ------------------------------ #


def save_processed_data(data, model_name, is_encoded=False):
    """
        Save the processed data to a CSV file.
        If the data is encoded, the file name will include '_encoded'.
    
        Args:
            data (pd.DataFrame): The processed data to save.
            model_name (str): The name of the model for which the data is being saved.
            is_encoded (bool): Whether the data is encoded. Default is False.

        Returns:
            str: The path to the saved file.
    """
    if is_encoded:
        model_name += '_encoded'
    save_path = os.path.join(DATA_PATH, 'processed', f"{model_name}_data.csv")

    os.makedirs(os.path.join(DATA_PATH, 'processed'), exist_ok=True)
    
    if os.path.exists(save_path):
        os.remove(save_path)

    data.to_csv(save_path, index=False)
    print(f"Dati salvati in: {save_path}")
    return save_path


# ------------------------------ Load Codons - AA Json ------------------------------ #

def load_codons_aa_json():
    """
        Load the codons to amino acids mapping from the JSON file.
    
        Returns:
            dict: The mapping of codons to amino acids.
    """
    file_path = os.path.join(PFAM_PATH, 'codons_aa.json')
    
    with open(file_path, 'r') as file:
        return json.load(file)
    

# ------------------------------ Load Processed Data ------------------------------ #

def load_processed_data(model_name, is_encoded=False):
    """
        Load the processed data from a CSV file.
        If the data is encoded, the file name will include '_encoded'.
    
        Args:
            model_name (str): The name of the model for which the data is being loaded.
            is_encoded (bool): Whether the data is encoded. Default is False.

        Returns:
            pd.DataFrame: The processed data loaded from the CSV file.
    """
    if is_encoded:
        model_name += '_encoded'
    file_path = os.path.join(DATA_PATH, 'processed', f"{model_name}_data.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato in: {file_path}")

    return pd.read_csv(file_path)