"""
Utility functions for path management and project configuration.
"""
from pathlib import Path

# Get the project root directory (nn-mar)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define common paths
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_OUTPUT = DATA_DIR / "output"
SRC_DIR = PROJECT_ROOT / "src"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PYTHON_FILES_DIR = PROJECT_ROOT / "python_files"

# Common subdirectories
FEATURES_DIR = DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"
WORLD_FEATURES_DIR = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
INPUT_FEATURES_DIR = DATA_INTERIM / "inputfeatures"
PREDICTION_MAPS_DIR = DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization"
FIGS_DIR = DATA_DIR / "figs"

def get_project_root():
    """Return the project root directory."""
    return PROJECT_ROOT

def get_data_dir():
    """Return the data directory."""
    return DATA_DIR

def get_src_dir():
    """Return the src directory."""
    return SRC_DIR

