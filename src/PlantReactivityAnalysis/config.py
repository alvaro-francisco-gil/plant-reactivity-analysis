from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
WAV_FOLDER = RAW_DATA_DIR / 'wav_files'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
EXTERNAL_DATA_DIR = DATA_DIR / 'external'

# Model directory
MODELS_DIR = BASE_DIR / 'models'

# Report directories
REPORTS_DIR = BASE_DIR / 'reports'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Notebook directory
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# References and documentation
REFERENCES_DIR = BASE_DIR / 'references'
DOCS_DIR = BASE_DIR / 'docs'
