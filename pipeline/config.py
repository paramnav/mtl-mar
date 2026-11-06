"""
Central configuration for the mtl-mar automation pipeline.

All paths, hyperparameter grids, model settings, and plotting defaults live
here so that every stage of the pipeline reads from a single source of truth.
Nothing in the existing codebase is modified — this package is additive.
Outputs go to pipeline/output/.
"""

from pathlib import Path

import numpy as np

# ── Project root (portable) ─────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]

# ── Raw data ────────────────────────────────────────────────────────────────
RAW_LABELS_DIR = ROOT / "data" / "raw" / "labels"
RAW_FEATURES_DIR = (
    ROOT / "data" / "raw" / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"
)
FEATURE_SELECTION_FILE = (
    ROOT / "data" / "interim" / "selectedFeatureLists" / "selectedfeatures_v_men.txt"
)

# Label CSV files  (Latitude, Longitude, target column)
LABEL_FILES = {
    "MAR_CS": {
        "csv": RAW_LABELS_DIR / "sedrate_continentalshelves.csv",
        "target_col": "MAR(g/cm**2/ka)",
        "keep_cols": ["Latitude", "Longitude", "MAR(g/cm**2/ka)"],
    },
    "MAR_DO": {
        "csv": RAW_LABELS_DIR / "sedrate_deep.csv",
        "target_col": "MAR(g/cm**2/ka)",
        "keep_cols": ["Latitude", "Longitude", "MAR(g/cm**2/ka)"],
    },
    "TOC_CS": {
        "csv": RAW_LABELS_DIR / "toc_continentalshelves_v2.csv",
        "target_col": "TOC [%]",
        "keep_cols": ["Latitude", "Longitude", "TOC [%]"],
    },
    "TOC_DO": {
        "csv": RAW_LABELS_DIR / "toc_deep_v2.csv",
        "target_col": "TOC [%]",
        "keep_cols": ["Latitude", "Longitude", "TOC [%]"],
    },
}

# ── Interim / processed data ───────────────────────────────────────────────
INTERIM_INPUT_FEATURES_DIR = ROOT / "data" / "interim" / "inputfeatures"
LAND_MAP_FILE = ROOT / "data" / "raw" / "island_map.npy"
N_ORIG_FEATURES = 113


def _first_existing(*candidates: Path) -> Path:
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


# Prefer NN-imputed masked world cubes when present (paper); else legacy men.
WORLD_FEATURES_DIR = _first_existing(
    ROOT / "data" / "interim" / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men_masked",
    ROOT / "data" / "interim" / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men",
)

# ── Pipeline outputs (separate from existing outputs) ──────────────────────
PIPELINE_OUTPUT_DIR = ROOT / "pipeline" / "output"
PIPELINE_MAPS_DIR = PIPELINE_OUTPUT_DIR / "maps"
PIPELINE_SCATTER_DIR = PIPELINE_OUTPUT_DIR / "scatter"
PIPELINE_HP_DIR = PIPELINE_OUTPUT_DIR / "hyperparameter_optimization"
PIPELINE_RESULTS_DIR = PIPELINE_OUTPUT_DIR / "results"
PIPELINE_MODELS_DIR = PIPELINE_OUTPUT_DIR / "models"
AUGMENTED_LABELS_DIR = PIPELINE_OUTPUT_DIR / "augmented_labels"

# ── Grid resolution (5 arc-minute) ─────────────────────────────────────────
N_LAT = 2160
N_LON = 4320
CHUNK_SHAPE = (6, 4320)  # each world feature file is 6 lat rows × 4320 lon cols
N_CHUNKS = N_LAT // CHUNK_SHAPE[0]  # 360

# ── Train / test split ─────────────────────────────────────────────────────
TEST_SIZE_RATIO = 0.25
RANDOM_SEED = 42
N_STRATIFY_BINS = 10

# ── Tree-based model hyperparameter sweeps ──────────────────────────────────
TREE_N_ESTIMATORS_SWEEP = np.concatenate(
    [
        np.linspace(1, 10, 10, dtype=int),
        np.linspace(20, 100, 9, dtype=int),
        np.linspace(200, 1000, 9, dtype=int),
    ]
)

KNN_K_SWEEP = list(range(1, 31)) + list(range(40, 101, 10))

# Paper Table-2 defaults (used when --skip-hp-sweep is set)
BEST_RF_N = 700
BEST_XGB_N = 1000
BEST_CB_N = 1000
BEST_KNN_K = 6

# ── DNN settings (paper-aligned FFNN) ───────────────────────────────────────
DNN_LAYER_WIDTHS = [128, 128, 128]
DNN_BATCH_SIZE = 64
DNN_EPOCHS = 1800
DNN_LR = 1e-4
DNN_WEIGHT_DECAY = 1e-4

# Paper attention defaults (FFNN + attention)
ATTENTION_CFG = dict(d_model=256, n_heads=8, n_layers=3, ff_mult=2, dropout=0.2)

# ── MC-Dropout for uncertainty ──────────────────────────────────────────────
MC_DROPOUT_SAMPLES = 100

# ── Plotting ────────────────────────────────────────────────────────────────
MAP_DPI = 200
SCATTER_DPI = 200
MAP_FIGSIZE = (25, 15)
SCATTER_FIGSIZE = (7, 7)
MAP_VMIN_MAR = 1e-1
MAP_VMAX_MAR = 1e3
MAP_CMAP_MAR = "cubehelix"
MAP_VMIN_TOC = 0
MAP_VMAX_TOC = 5
MAP_CMAP_TOC = "viridis"
