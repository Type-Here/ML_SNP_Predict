import os
import tempfile
import subprocess

from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QFrame, QTextEdit, QLabel, QWidget
from Bio.PDB import PDBList
from PyQt5 import Qt
from PyQt5.QtCore import QUrl

from src.config import PFAM_PATH, P53_PDB, HRAS_PDB, HTML_NGL_PATH


def add_protein_3d_view(biopy_layout:QVBoxLayout, text_edit:QTextEdit, pdb_code = P53_PDB):
    # Get the PDB file path
    pdb_file = __get_pdb_file(pdb_code, text_edit)
    if pdb_file is None:
        print("PDB file not found.")
        text_edit.append("PDB file not found: " + pdb_code + ".\n Unable to display the 3D structure.")
        return False
    
    output_image = "protein_view.png"

    # Generate the image using PyMOL
    generate_pymol_image(pdb_file, output_image)

    # Create the ProteinViewer widget to display the image
    viewer = ProteinViewer(output_image)
    
    # Add the viewer to the layout
    biopy_layout.addWidget(viewer)

    return True


def __get_pdb_file(pdb_code, text_edit):
    """
        Get the PDB file path for the given PDB code.
        The PDB file is searched in the PFAM_PATH directory.
        If the file is not found it tries to download it.
        Parameters:
            pdb_code: The PDB code of the protein.
            text_edit: The QTextEdit widget to display messages.
        Returns:
            str: The path to the PDB file or None if the file is not found.
    """

    # Get the PDB file path
    pdb_file = os.path.join(PFAM_PATH, pdb_code + ".pdb")
    if not os.path.exists(pdb_file):
        text_edit.append("PDB file not found. Trying to download it.")
        pdb_file = __download_pdb(pdb_code, text_edit)
        return pdb_file
    
    pdb_file = os.path.join(PFAM_PATH, pdb_code + ".pdb")
    return pdb_file


def __download_pdb(pdb_code, text_edit):
    """
        Download the PDB file for the given PDB code.
        The PDB file is downloaded from the RCSB PDB database using Biopython.
        Parameters:
            pdb_code: The PDB code of the protein.
            text_edit: The QTextEdit widget to display messages.
        Returns:
            str: The path to the downloaded PDB file or None if the download failed.
    """
    # Download the PDB file
    pdb = PDBList()
    file_path = pdb.retrieve_pdb_file(pdb_code, file_format='pdb', pdir=PFAM_PATH)

    # Check if the download was successful
    if file_path is None:
        if text_edit is not None:
            text_edit.append("PDB file download failed.")
        return None
    
    return file_path



# -------------------------------------------- PYMOL -------------------------------------------- #

def generate_pymol_image(pdb_file, output_image):
    pymol_script = f"""
    load {pdb_file}
    show cartoon
    color cyan
    png {output_image}, dpi=300
    quit
    """
    with open("pymol_script.pml", "w") as script_file:
        script_file.write(pymol_script)
    subprocess.run(["pymol", "-cq", "pymol_script.pml"])


    # -------------------------------------------- CLASS ProteinViewer -------------------------------------------- #


class ProteinViewer(QWidget):
    def __init__(self, image_path):
        super().__init__()
        self.initUI(image_path)

    def initUI(self, image_path):
        layout = QVBoxLayout()
        label = QLabel()
        pixmap = QPixmap(image_path)
        label.setPixmap(pixmap)
        layout.addWidget(label)
        self.setLayout(layout)
        self.setWindowTitle("Protein Viewer")