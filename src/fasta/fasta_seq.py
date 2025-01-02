"""
    This module contains the functions to download fasta DNA sequences from NCBI.
"""

import requests
import os
from src.config import FASTA_PATH

def download_fasta(nm_code, file_name, save_dir=FASTA_PATH):
    """
    Download a FASTA sequence from NCBI and save it to a local file.
    
    Parameters:
        nm_code (str): NM - NG Code of the sequence (ex: 'NM_000546.6').
        file_name (str): file name without extension (ex: 'p53_sequence').
        save_dir (str): Directory in which to save the file (default: 'FASTA_PATH').
    
    Returns:
        str: The path of the saved file.
    """
    # URL di download
    url = f"https://www.ncbi.nlm.nih.gov/nuccore/{nm_code}?report=fasta&log$=seqview&format=text"
    
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_name}.fasta")
    
    try:
        # Download the file
        print(f"Scaricamento da: {url}")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        # Save the file
        with open(file_path, 'w') as file:
            file.write(response.text)
        
        print(f"File salvato in: {file_path}")
        return file_path
    
    except requests.exceptions.RequestException as e:
        print(f"Errore durante il download: {e}")
        return None


def read_fasta(file_path):
    """
    Read a FASTA file and return the sequence.
    
    Parameters:
        file_path (str): The path to the FASTA file.
    
    Returns:
        str: The DNA sequence.
    """
    try:
        with open(file_path, 'r') as file:
            # Skip the first line (header)
            file.readline()
            # Read the sequence
            sequence = file.read().replace('\n', '')
            return sequence
    except OSError as e:
        print(f"Errore durante la lettura del file: {e}")
        return None    
    