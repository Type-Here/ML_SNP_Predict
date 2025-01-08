import os


# Global variables
# Base directory (assoluto e normalizzato)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Data directory path
DATA_PATH = os.path.normpath(os.path.join(BASE_DIR, 'data'))

# Fasta files directory path
FASTA_PATH = os.path.normpath(os.path.join(DATA_PATH, 'sequences'))

# Pfam directory path
PFAM_PATH = os.path.normpath(os.path.join(DATA_PATH, 'protein'))

# Models directory path
MODELS_DIR = os.path.normpath(os.path.join(BASE_DIR, 'models'))

# Models stats directory path
MODELS_STATS_DIR = os.path.normpath(os.path.join(MODELS_DIR, 'stats'))

# Plots save directory path
PLOTS_SAVE_DIR = os.path.normpath(os.path.join(MODELS_DIR, 'plots'))

# p53 variables
P53_MODEL_NAME = 'p53_model'
P53_PFAM_MODEL_NAME = 'p53_pfam_model'

P53_NM = 'NM_000546.6'
P53_ACCESSION = 'P04637'
P53_PDB = '1TUP'

# HRAS variables
HRAS_MODEL_NAME = 'hras_model'

HRAS_NM = 'NM_005343.2'
HRAS_ACCESSION = 'P01112'
HRAS_PDB = '5P21'

