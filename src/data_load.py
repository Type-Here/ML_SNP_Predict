import os
import pandas as pd
import requests
import zipfile
from src.config import DATA_PATH

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
            return _ensure_p53_data()

        case _: 
            ValueError(f"Protein {protein} not supported.")




# Internal function to ensure p53 data is available
def _ensure_p53_data():
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