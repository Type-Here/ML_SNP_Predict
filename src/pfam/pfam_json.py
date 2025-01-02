"""
    This module contains the functions to parse the Pfam JSON file.
"""

import requests
import json
from src.config import PFAM_PATH


# Function to download Pfam data of a protein
def download_pfam_domains(protein_accession):
    """
        Download the Pfam domains for a protein from the EBI InterPro API.
        Parameters:
            protein_accession (str): The UniProt accession number of the protein.
        Returns:
            dict: The Pfam data.
    """
    url = f"https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/{protein_accession}/"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Errore nel download dei domini Pfam per {protein_accession}: {response.status_code}")
        return None
    

def save_pfam_json(pfam_data, dir_path = PFAM_PATH, file_name = "pfam_data.json"):
    """
        Save the Pfam data to a JSON file.
        Parameters:
            pfam_data (dict): The Pfam data to save.
            dir_path (str): The directory path of the file to save the data to.
            file_name (str): The name of the file to save the data to.
    """
    file_path = f"{dir_path}/{file_name}"
    with open(file_path, 'w') as file:
        json.dump(pfam_data, file)
    print(f"File salvato in: {file_path}")


def load_pfam_json(file_path = PFAM_PATH, file_name = "pfam_data.json"):
    """
        Load the Pfam data from a JSON file.
        Parameters:
            file_path (str): The directory path of the file to load the data from.
            file_name (str): The name of the file to load the data from.
        Returns:
            dict: The Pfam data.
    """
    file_path = f"{file_path}/{file_name}"
    with open(file_path, 'r') as file:
        return json.load(file)


# Function to create list of dicts of Pfam domains
def create_domain_dict(pfam_data):
    """
        Create a list of dictionaries containing the domain information.
        Parameters:
            pfam_data (dict): The Pfam data.
        Returns:
            list[dict]: A list of dictionaries containing the domain information.
    """
    domain_list = []
    for result in pfam_data.get("results", []):
        for protein in result.get("proteins", []):
            for location in protein.get("entry_protein_locations", []):
                score = location.get("score", 0)  # Get the score from the location level
                for fragment in location.get("fragments", []):
                    domain_map = {
                        "start": fragment["start"], 
                        "end" : fragment["end"],
                        "score": score
                    }
                    domain_list.append(domain_map)
    return domain_list


# Function to assign domain and conseravtion value
def assign_conservation(row, domain_dict):
    """
        Assign a conservation score based on the position in the protein.
        Parameters:
            row (pd.Series): The row of the DataFrame.
            domain_dict (list[dict]): A list of dictionaries containing the domain information.
        Returns:
            int: The conservation score.
    """
    position = row["Intron"]
    if position == True:
        return 0  # Introns

    for map in domain_dict:
        start, end, score = map["start"]*3, map["end"]*3, map["score"]
        if start <= position <= end:
            return score
    return 0  # Unknown mutations