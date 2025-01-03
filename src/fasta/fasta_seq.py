"""
    This module contains the functions to download fasta DNA sequences from NCBI.
"""

import requests
import os
from src.config import FASTA_PATH, P53_NM, HRAS_NM
from Bio import SeqIO, Entrez

def get_fasta_sequence_by_model_name(name:str, email:str):
    """
    Get the FASTA sequence by the name of the model.
    Try to load the sequence from a local file, if it doesn't exist download it from NCBI.
    See get_fasta_sequence for more details.
    """
    nm_code = None
    match name:
        case "P53 Model":
            nm_code = P53_NM
        case "P53 Pfam":
            nm_code = P53_NM
        case "Hras Transfer":
            nm_code = HRAS_NM
        case _:
            return None
    
    if nm_code:
        return get_fasta_sequence(nm_code, email)
    else:
        return None



def get_fasta_sequence(nm_code:str, email:str) -> str:
    """
    Get the FASTA sequence. \n
    Try to load the sequence from a local file, if it doesn't exist download it from NCBI.

    Parameters:
        nm_code (str): NM - NG Code of the sequence (ex: 'NM_000546.6').

    Returns:
        str: The DNA sequence.
    """
    
    if not nm_code:
        return None
    

    file_name = nm_code.replace('.', '_')
    file_path = os.path.join(FASTA_PATH, f"{file_name}.fasta")
    
    try:
        # Try to read the file
        sequence = __read_fasta(file_path)
        if sequence:
            return sequence
    except FileNotFoundError:
        # If the file does not exist, download it
        file_path = __download_fasta_entrez(nm_code, file_name, email)
        if file_path:
            return __read_fasta(file_path)
    
    return None

@DeprecationWarning
def __download_fasta(nm_code, file_name, save_dir=FASTA_PATH):
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


def __read_fasta(file_path):
    """
    Read a FASTA file and return the sequence.
    
    Parameters:
        file_path (str): The path to the FASTA file.
    
    Returns:
        str: The DNA sequence.
    """
    with open(file_path, 'r') as file:
        # Skip the first line (header)
        file.readline()
        # Read the sequence
        sequence = file.read().replace('\n', '')
        return sequence
    

def __download_fasta_entrez(nm_code, file_name, email, save_dir=FASTA_PATH):
    """
    Download a FASTA sequence from NCBI using Entrez and save it to a local file.
    
    Parameters:
        nm_code (str): NM - NG Code of the sequence (ex: 'NM_000546.6').
        file_name (str): file name without extension (ex: 'p53_sequence').
        email (str): Un'email richiesta da Entrez per identificazione.
        save_dir (str): Directory in which to save the file (default: 'FASTA_PATH').

    Returns:
        str: Path of the saved file.
    """
    # Set the email for Entrez
    Entrez.email = email
    
    # Create the save directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{file_name}.fasta")
    
    try:
        # Download the sequence from NCBI
        print(f"Scaricamento della sequenza {nm_code} da NCBI...")
        with Entrez.efetch(db="nucleotide", id=nm_code, rettype="fasta", retmode="text") as handle:
            # Save the file
            with open(file_path, 'w') as file:
                file.write(handle.read())
        
        print(f"Sequenza salvata in: {file_path}")
        return file_path
    
    except Exception as e:
        print(f"Errore durante il download: {e}")
        return None
