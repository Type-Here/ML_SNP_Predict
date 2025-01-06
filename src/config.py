import os


# Global variables

# Data directory path
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/')
# Fasta files directory path
FASTA_PATH = os.path.join(os.path.dirname(__file__), '../data/sequences/')
# Pfam directory path
PFAM_PATH = os.path.join(os.path.dirname(__file__), '../data/protein/')
# Models directory path
MODELS_DIR = os.path.join(os.path.dirname(__file__), '../models/')
MODELS_STATS_DIR = os.path.join(os.path.dirname(__file__), '../models/stats/')
PLOTS_SAVE_DIR = os.path.join(os.path.dirname(__file__), '../models/plots/')

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

