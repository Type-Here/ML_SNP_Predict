import os


# Global variables

# Data directory path
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')
# Fasta files directory path
FASTA_PATH = os.path.join(os.path.dirname(__file__), '../data/sequences/')
# Pfam directory path
PFAM_PATH = os.path.join(os.path.dirname(__file__), '../data/protein/')

# p53 variables
P53_MODEL_NAME = 'p53_model'
P53_MODEL_DIR = os.path.join(os.path.dirname(__file__), '../models/')

P53_NM = 'NM_000546.6'
P53_ACCESSION = 'P04637'

# HRAS variables
HRAS_NM = 'NM_005343.2'
HRAS_ACCESSION = 'P01112'


