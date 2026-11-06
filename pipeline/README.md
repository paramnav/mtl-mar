# Automation pipeline — new data → new results

Ported from `nn4sedratetoc/pipeline/` for **mtl-mar**. Does not modify existing
notebooks or training code; all outputs go to `pipeline/output/`.

For the **full paper model suite** (TabPFN, MTL, attention, …) and NN-imputation,
prefer `docs/RETRAINING_WITH_NEW_DATA.md` and:

```bash
bash notebooks/run_all_models.sh
```

This `pipeline/` path is the lightweight route (trees / kNN / optional DNN + HP sweeps).

## Add new observations

```bash
# Append CSV (columns: Latitude, Longitude, MAR(g/cm**2/ka) or TOC [%])
python pipeline/add_data.py --dataset MAR_CS --csv new_points.csv

# Single point
python pipeline/add_data.py --dataset MAR_CS --lat -38.5 --lon 178.2 --value 120.5

# Interactive
python pipeline/add_data.py --dataset MAR_CS --interactive

# Summary
python pipeline/add_data.py --dataset MAR_CS --summary
```

Datasets: `MAR_CS`, `MAR_DO`, `TOC_CS`, `TOC_DO`.  
Augmented labels are stored in `pipeline/output/augmented_labels/` (originals untouched).

## Run the pipeline

```bash
# Full rebuild (extract features from raw rasters + train trees)
python pipeline/run_pipeline.py

# Use existing interim features (fast); paper tree HPs; skip maps
python pipeline/run_pipeline.py --use-existing-features --skip-hp-sweep --skip-maps

# MAR only, RF + XGBoost
python pipeline/run_pipeline.py --target MAR --models rf xgb --use-existing-features

# Include FFNN or attention DNN
python pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch mlp
python pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch attention
```

SLURM:

```bash
sbatch pipeline/slurm_pipeline.sbatch
```

## Steps

| Step | Script | Role |
|------|--------|------|
| 1 | `step1_build_features.py` | Extract site features at new/augmented labels |
| 2 | `step2_train_and_evaluate.py` | RF / XGB / CatBoost / KNN + scatters + maps |
| 3 | `step3_train_dnn.py` | FFNN (`mlp`) or feature-token `attention` |

Config: `pipeline/config.py` (paths, paper defaults RF=700, KNN k=6, FFNN `[128]×3`).

## Outputs

```
pipeline/output/
  features/              # rebuilt NumPy arrays per dataset
  scatter/               # train/test scatters
  hyperparameter_optimization/
  maps/                  # global .npy + .png
  models/                # DNN weights
  results/               # JSON metrics
  augmented_labels/      # only if add_data.py was used
```

To revert new data: delete the augmented CSV under `augmented_labels/` and re-run.
