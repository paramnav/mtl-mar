# Retraining with new MAR/TOC data or new features

This is the detailed operator guide for **mtl-mar** (GitHub: [paramnav/mtl-mar](https://github.com/paramnav/mtl-mar)).  
A shorter summary appears in the paper Supporting Information (automation / reproducibility section).

There are **two** related paths:

| Path | When to use | Entry point |
|------|-------------|-------------|
| **A. Paper workflow (`notebooks/`)** | Reproduce or update *all* manuscript models (RF, XGB, CatBoost, kNN, TabPFN, FFNN, MTL, attention) and maps | `notebooks/run_all_models.sh` |
| **B. Lightweight pipeline (`pipeline/`)** | Quickly add points, rebuild site features, tree/KNN (+ optional FFNN) with HP sweeps into `pipeline/output/` | `python pipeline/run_pipeline.py` |

Do **not** mix them for one analysis: Path A writes under `data/output/` via the train notebooks; Path B isolates outputs under `pipeline/output/`.

---

## A — Full paper workflow (recommended for new global maps)

### A.1 Checklist when labels or feature rasters change

1. **Labels** — `notebooks/collect_SR_MAR_datapoints.ipynb`  
   Rebuild CS/DO MAR (and TOC if needed) observation tables.

2. **Site features** —  
   `create_training_features_CS.ipynb`, `create_training_features_DO.ipynb`  
   Optional edits: `modify_features.ipynb`.

3. **Global feature grids** — `Make_Global_Features.ipynb`  
   Resample every environmental layer onto the common 5′×5′ ocean grid (including any new rasters you added under `data/raw/features/`).

4. **NN ocean-gap imputation** — fill product NaNs / polar gaps *before* training:

   ```bash
   # GPU recommended; multi-hour
   python scripts/harmonize_ocean_gaps.py
   python scripts/build_masked_model_features.py
   ```

   Or run `notebooks/nn_feature_cross_imputation.ipynb` with  
   `RUN_HARMONIZE=True` and `RUN_BUILD_PRODUCTS=True`.  
   Details: `scripts/README_NN_IMPUTATION.md`.

   Products land in:
   - `data/interim/inputfeatures/SedTOCFeaturesnoNAN_*_men_masked/`
   - `data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked/`

5. **Train all models + write maps** — one command:

   ```bash
   cd /path/to/mtl-mar
   bash notebooks/run_all_models.sh            # SLURM: submit all jobs
   bash notebooks/run_all_models.sh --list      # dry-run
   bash notebooks/run_all_models.sh --local     # sequential on current node
   bash notebooks/run_all_models.sh --trees-only
   bash notebooks/run_all_models.sh --gpu-only
   ```

   Individual legacy batchfiles remain under `notebooks/batchfiles/`.  
   Interactive guide: `notebooks/run_models_on_new_data.ipynb`.

### A.2 Hyperparameter search

- **Selected Table-2 configs** are the defaults in every `train_*.ipynb` (and in `run_all_models.sh`).  
  After a successful run you get scatters + prediction maps under `data/output/`.

- **Tree / kNN sweeps** (n_estimators, k, …): use Path B without `--skip-hp-sweep`, or set  
  `run_hyperparameter_search = True` inside `train_tree_based_models.ipynb` / `train_knn.ipynb` before executing.

- **Deep-learning architecture grids** (FFNN / MTL / attention) are multi-day GPU searches.  
  Paper Text~S3 and the embedded notebook outputs document the selected widths / shared-layer counts.  
  Re-running those grids is optional research work, not required to refresh maps with new data at the *selected* architectures.

### A.3 Reproduce manuscript Table 2 *without* new data

Same checkpoints / SI maps as the paper (no retraining from scratch):

```bash
sbatch notebooks/batchfiles/batchfile_reproduce_paper_table2
# or
python notebooks/reproduce_paper_table2.py
```

---

## B — Lightweight automation pipeline

Does **not** edit raw CSVs. Augmented labels go to `pipeline/output/augmented_labels/`.

```bash
# Add points
python pipeline/add_data.py --dataset MAR_CS --csv new_points.csv

# Rebuild features, HP sweeps, tree/KNN (+ optional DNN), maps
python pipeline/run_pipeline.py
python pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch mlp
sbatch pipeline/slurm_pipeline.sbatch
```

See `pipeline/README.md` for flags and output layout.

---

## Feature width reminder

| Model family | Predictors |
|--------------|------------|
| Trees, kNN, TabPFN, FFNN, MTL | 113 NN-imputed values (`MASKED_FEATURE_VARIANT=values_only`) |
| Attention FFNN / MTL | 226 = 113 values + 113 missingness masks (`attention`) |

`run_all_models.sh` sets the variant per job automatically.

---

## Typical “new cores → new global MAR map” day

```bash
# 1–4: rebuild labels/features/imputation (notebooks or scripts above)
# 5: train everything at paper HPs
bash notebooks/run_all_models.sh

# watch GPU jobs
tail -f notebooks/outfiles/train_*_*.out
```

Inspect maps and metrics under `data/output/`, then update paper figures if the new maps are adopted.
