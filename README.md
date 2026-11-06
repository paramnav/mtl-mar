# MTL-MAR: Global Prediction of Mass Accumulation Rates Using Multi-Task Learning

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18591773-blue)](https://doi.org/10.5281/zenodo.18591773)

**Companion code for:**
> Parameswaran, N., Burwicz-Galerne, E., Gonzalez, E., Braack, M., & Wallmann, K.
> *Global Prediction of Mass Accumulation Rates of Sediments and Organic Carbon Turnover Using Multi-Task Learning.*
> Global Biogeochemical Cycles (2026).

This repository trains and evaluates machine-learning models that predict
**marine mass accumulation rate (MAR)** and **total organic carbon (TOC)**
in surface sediments from a stack of 113 environmental features on a
global 5′ × 5′ ocean grid. The selected model is a **multi-task learning
(MTL) neural network** that jointly predicts MAR and TOC with hard
parameter sharing; tree-based models, k-NN, TabPFN, feed-forward neural
networks (FFNN), and attention-based tabular encoders are included for
comparison.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Quick Start](#quick-start)
3. [Data Requirements](#data-requirements)
4. [Workflow Overview](#workflow-overview)
5. [Training Models](#training-models)
6. [Key Figures](#key-figures)
7. [Retraining with New Data](#retraining-with-new-data)
8. [Lightweight Pipeline](#lightweight-pipeline)
9. [Reproducibility Notes](#reproducibility-notes)
10. [License](#license)

---

## Repository Structure

```
mtl-mar/
│
├── notebooks/                        # Jupyter notebooks — the main entry point
│   ├── collect_SR_MAR_datapoints.ipynb        # Step 1: Compile MAR/TOC labels
│   ├── create_training_features_CS.ipynb      # Step 2a: Site features (continental shelf)
│   ├── create_training_features_DO.ipynb      # Step 2b: Site features (deep ocean)
│   ├── modify_features.ipynb                  # Step 2c: Optional feature edits
│   ├── Make_Global_Features.ipynb             # Step 3: Global 5′×5′ feature grids
│   ├── nn_feature_cross_imputation.ipynb      # Step 4: NN ocean-gap imputation
│   ├── train_tree_based_models.ipynb          # Step 5: RF, XGBoost, CatBoost
│   ├── train_knn.ipynb                        # Step 5: k-Nearest Neighbors
│   ├── train_tabpfn.ipynb                     # Step 5: TabPFN
│   ├── train_neural_network.ipynb             # Step 5: Feed-forward NN (FFNN)
│   ├── train_multi_task_learning.ipynb        # Step 5: MTL (selected model)
│   ├── train_neural_network_attention.ipynb   # Step 5: FFNN + attention
│   ├── train_multi_task_learning_attention.ipynb  # Step 5: MTL + attention
│   ├── run_models_on_new_data.ipynb           # Interactive checklist for retraining
│   ├── run_all_models.sh                      # One-command SLURM submission for all models
│   ├── reproduce_paper_table2.py              # Reinstall archived Table 2 products
│   ├── paper_train_runtime.py                 # Shared training/eval helpers
│   ├── paper_results_display.py               # Map and figure plotting helpers
│   ├── embed_hp_results_into_notebooks.py     # Embed HP search outputs into notebooks
│   ├── batchfiles/                            # Individual SLURM job scripts
│   └── README.md                              # Notebook-specific workflow details
│
├── scripts/                          # Standalone Python scripts
│   ├── harmonize_ocean_gaps.py                # Driver: NN imputation of ocean NaNs
│   ├── build_masked_model_features.py         # Build training arrays with fill indicators
│   ├── nn_feature_imputation.py               # Per-feature MLP imputer core
│   ├── ml_feature_imputation.py               # Shared cube/native-mask helpers
│   ├── spatial_interpolation.py               # Polar IDW / nearest fallback
│   ├── mask_gap_core.py                       # Gap detection and mask I/O
│   ├── feature_update_core.py                 # Feature stack paths + r50km helpers
│   ├── build_interpolation_masks.py           # Fill-indicator stack
│   ├── recompute_feature_stats.py             # Native-ocean mean/std
│   ├── rebuild_harmonized_feature_products.py # Site arrays + world chunks
│   └── README_NN_IMPUTATION.md                # NN imputation documentation
│
├── src/                              # Reusable Python library modules
│   ├── preprocessing/
│   │   ├── makeTrainingFeatures.py            # Feature extraction for training sites
│   │   ├── makeGlobalFeatures.py              # Global feature grid generation
│   │   ├── masked_features.py                 # Feature loading with mask variants
│   │   ├── nn_imputed_features.py             # NN-imputed feature path resolution
│   │   └── men_r50km.py                       # Spatial averaging (50 km radius)
│   ├── models/
│   │   ├── train_neural_network.py            # FFNN training + evaluation
│   │   ├── train_multi_task_learning.py       # MTL training + evaluation
│   │   ├── train_tree_based_models.py         # Tree model wrappers
│   │   ├── tree_based_models.py               # RF/XGB/CatBoost core
│   │   ├── predict_neural_network.py          # NN prediction helpers
│   │   ├── explain_multi_task_learning.py     # SHAP / feature importance (MTL)
│   │   └── explain_neural_network.py          # SHAP / feature importance (FFNN)
│   ├── postprocessing/
│   │   ├── evaluate_model_performance.py      # Metrics (MAE, CC, R², budgets)
│   │   ├── visualize_prediction_map.py        # Map plotting (Robinson projection)
│   │   ├── convert_to_xarray.py               # Export to NetCDF / xarray
│   │   └── curve_fitting.py                   # Distribution fitting
│   ├── download_data_from_zenodo.py           # Zenodo data download
│   └── utils.py                               # Shared utilities
│
├── pipeline/                         # Lightweight automation pipeline
│   ├── add_data.py                            # Append new MAR/TOC observations
│   ├── run_pipeline.py                        # One-command: features → train → maps
│   ├── step1_build_features.py                # Extract site features
│   ├── step2_train_and_evaluate.py            # Tree/kNN training + evaluation
│   ├── step3_train_dnn.py                     # Optional DNN (MLP or attention)
│   ├── config.py                              # Paths, paper defaults, HP grids
│   ├── slurm_pipeline.sbatch                  # SLURM job wrapper
│   └── README.md                              # Pipeline flags and usage
│
├── data/                             # All data (raw, interim, output) — gitignored
│   ├── raw/
│   │   ├── features/                          # Environmental raster layers (from Zenodo)
│   │   ├── labels/                            # MAR/TOC observation CSVs
│   │   └── island_map.npy                     # Land/ocean mask
│   ├── interim/
│   │   ├── inputfeatures/                     # Site-level training arrays
│   │   ├── worldfeatures/                     # Global prediction grid chunks
│   │   └── masks/                             # Interpolation/gap masks
│   └── output/
│       ├── predictionmaps_SedRate/            # Global MAR maps (.npy + .png)
│       ├── model_metrics/                     # JSON / CSV metrics
│       └── nc/                                # NetCDF exports
│
├── docs/
│   └── RETRAINING_WITH_NEW_DATA.md            # Full operator guide
│
├── _archive/                         # Legacy code preserved for provenance
│   ├── notebooks_v1/                          # Original notebook versions
│   └── python_files/                          # Standalone training scripts (superseded)
│
├── setup_project.py                  # Complete environment setup
├── requirements.txt                  # Python dependencies
├── README_DATA_DOWNLOAD.md           # Zenodo download instructions
└── .gitignore
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/paramnav/mtl-mar.git
cd mtl-mar
pip install -r requirements.txt
```

### 2. Download data from Zenodo

Raw features and labels must be downloaded before running any notebooks.
See [`README_DATA_DOWNLOAD.md`](README_DATA_DOWNLOAD.md) for details.

```bash
python setup_project.py
```

### 3. Run the selected MTL model

```bash
# Option A: run a single notebook interactively
jupyter notebook notebooks/train_multi_task_learning.ipynb

# Option B: submit all paper models to SLURM
bash notebooks/run_all_models.sh
```

---

## Data Requirements

| Data | Source | Location after download |
|------|--------|------------------------|
| Environmental feature rasters (113 layers) | [Zenodo archive](https://doi.org/10.5281/zenodo.18591773) | `data/raw/features/` |
| MAR observation labels | Zenodo | `data/raw/labels/sedrate_continentalshelves.csv`, `sedrate_deep.csv` |
| Land/ocean mask | Zenodo | `data/raw/island_map.npy` |

Feature layers include bathymetry (ETOPO 2022), sediment thickness (GlobSed v3),
ocean hydrography (WOA23, HYCOM, MIMOC), ocean color (VIIRS/SNPP),
surface wind (ERA5), wave climate (WAVEWATCH III), tidal currents (EOT20),
lithology (Garlan, Diesing), river carbon fluxes (Liu et al. 2024),
biomass (Wei), and others. See the paper Supporting Information Table S1
for the complete list with native resolution and gap-fill method.

---

## Workflow Overview

The end-to-end pipeline has five stages, executed through the Jupyter
notebooks in `notebooks/`:

```
┌───────────────────────────────────────────────────────────────────┐
│  1. Collect labels              collect_SR_MAR_datapoints.ipynb   │
│  2. Build site features         create_training_features_{CS,DO}  │
│  3. Build global feature grids  Make_Global_Features.ipynb        │
│  4. NN ocean-gap imputation     scripts/harmonize_ocean_gaps.py   │
│                                 scripts/build_masked_model_features│
│  5. Train + evaluate + map      train_*.ipynb or run_all_models.sh│
└───────────────────────────────────────────────────────────────────┘
```

### Step-by-step

| Step | Notebook / Script | What it does |
|------|-------------------|-------------|
| 1 | `notebooks/collect_SR_MAR_datapoints.ipynb` | Compile MAR (and TOC) measurements from raw observation databases into site-level CSVs |
| 2a | `notebooks/create_training_features_CS.ipynb` | Extract the 113-feature vector at each continental-shelf MAR site |
| 2b | `notebooks/create_training_features_DO.ipynb` | Same for deep-ocean sites |
| 2c | `notebooks/modify_features.ipynb` | *(Optional)* add, remove, or rename features |
| 3 | `notebooks/Make_Global_Features.ipynb` | Resample all environmental layers onto the 5′ × 5′ prediction grid |
| 4a | `python scripts/harmonize_ocean_gaps.py` | Per-feature NN cross-imputation of ocean NaN gaps (GPU recommended) |
| 4b | `python scripts/build_masked_model_features.py` | Build training arrays (113 values + 113 fill indicators) |
| 5 | `notebooks/train_*.ipynb` | Train each ML model, produce scatter plots, global prediction maps, and metrics |

---

## Training Models

The following models are trained and compared in the paper:

| Model | Notebook | Selected hyperparameters |
|-------|----------|--------------------------|
| Random Forest | `train_tree_based_models.ipynb` | 700 trees |
| XGBoost | `train_tree_based_models.ipynb` | 1000 trees |
| CatBoost | `train_tree_based_models.ipynb` | 1000 trees |
| k-NN | `train_knn.ipynb` | k = 6 |
| TabPFN | `train_tabpfn.ipynb` | Pretrained baseline |
| FFNN | `train_neural_network.ipynb` | [128, 128, 128], dropout 0.2 |
| **MTL (selected)** | **`train_multi_task_learning.ipynb`** | **[128, 128, 128, 128], 2 shared layers** |
| FFNN + attention | `train_neural_network_attention.ipynb` | d256, 8 heads, 3 layers |
| MTL + attention | `train_multi_task_learning_attention.ipynb` | d128, 8 heads, 2 layers |

### Run all models at once

```bash
bash notebooks/run_all_models.sh            # submit all SLURM jobs
bash notebooks/run_all_models.sh --list     # dry-run: print job names
bash notebooks/run_all_models.sh --local    # run sequentially (no SLURM)
bash notebooks/run_all_models.sh --trees-only   # CPU models only
bash notebooks/run_all_models.sh --gpu-only     # GPU models only
```

### Feature variants

| Variant | Columns | Used by |
|---------|---------|---------|
| `values_only` | 113 NN-imputed values | Trees, kNN, TabPFN, FFNN, MTL |
| `attention` | 226 = 113 values + 113 missingness masks | FFNN+attention, MTL+attention |

The `run_all_models.sh` script sets the correct variant per model automatically.

---

## Key Figures

The notebooks and scripts produce the following key outputs under `data/output/`:

| Figure | Description | Location |
|--------|-------------|----------|
| Global MAR prediction maps | Robinson-projection maps for each model | `data/output/predictionmaps_SedRate/paper_table2/maps_paper_png/` |
| Train/test scatter plots | Measured vs. predicted MAR | `data/output/predictionmaps_SedRate/paper_table2/scatters/` |
| Hyperparameter optimization | Test MAE, CC, R² vs. tree count / architecture | `data/output/predictionmaps_SedRate/hyperparameter_optimization/` |
| MC-dropout uncertainty | Model uncertainty at each grid cell | Generated by MTL notebook |
| Information gain map | KL divergence identifying high-value sampling sites | Generated by MTL notebook |
| TOC comparison | MTL vs. nn-toc model differences | Generated by MTL notebook |

---

## Retraining with New Data

**Full guide:** [`docs/RETRAINING_WITH_NEW_DATA.md`](docs/RETRAINING_WITH_NEW_DATA.md)

### Short version (Path A — all paper models)

When you have new MAR observations or updated environmental rasters:

```bash
# 1. Rebuild labels (edit and run the collection notebook)
#    notebooks/collect_SR_MAR_datapoints.ipynb

# 2. Rebuild site features (run the CS and DO feature notebooks)
#    notebooks/create_training_features_CS.ipynb
#    notebooks/create_training_features_DO.ipynb

# 3. Rebuild global grids (if rasters changed)
#    notebooks/Make_Global_Features.ipynb

# 4. Re-run NN ocean-gap imputation (GPU recommended)
python scripts/harmonize_ocean_gaps.py
python scripts/build_masked_model_features.py

# 5. Train everything at paper hyperparameters
bash notebooks/run_all_models.sh

# 6. Inspect results
ls data/output/predictionmaps_SedRate/
```

### Reproduce manuscript Table 2 (same data, no retraining)

```bash
python notebooks/reproduce_paper_table2.py
```

---

## Lightweight Pipeline

For quick updates (trees/kNN + optional DNN) without running all model families:

```bash
# Append new observation points
python pipeline/add_data.py --dataset MAR_CS --csv new_points.csv

# Full pipeline: features → HP sweeps → train → maps
python pipeline/run_pipeline.py

# Faster: use existing features, skip HP sweep
python pipeline/run_pipeline.py --use-existing-features --skip-hp-sweep

# Include DNN
python pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch mlp

# SLURM
sbatch pipeline/slurm_pipeline.sbatch
```

| Flag | Description |
|------|-------------|
| `--target {MAR,TOC}` | Process only MAR or TOC (default: both) |
| `--models rf xgb cb knn` | Select tree/kNN models |
| `--skip-hp-sweep` | Use default hyperparameters |
| `--skip-maps` | Skip global map generation |
| `--use-existing-features` | Read from `data/interim/` |
| `--dnn` | Include DNN training |
| `--dnn-arch {mlp,attention}` | DNN architecture |
| `--mc-samples N` | MC-dropout samples (default: 100) |

Pipeline outputs go to `pipeline/output/` (separate from the main `data/output/`).

See [`pipeline/README.md`](pipeline/README.md) for details.

---

## Reproducibility Notes

- **Random seed:** All models use seed 42 for train/test splitting and weight initialization.
- **GPU dependence:** Neural network results were produced on NVIDIA H100 GPUs.
  TensorFloat-32 (TF32) is used by default on H100. For closer cross-GPU agreement,
  disable TF32:
  ```python
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  ```
- **Feature variant:** Set `MASKED_FEATURE_VARIANT=values_only` for standard models
  or `attention` for attention-based models. `run_all_models.sh` handles this automatically.

---

## License

This project is licensed under the **MIT License**.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
