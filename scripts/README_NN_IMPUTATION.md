# NN cross-feature imputation (ocean gaps)

Paper Text~S1 / methods: fill ocean feature gaps with a **per-feature MLP**,
using only **natively observed** predictors (imputed values are never chained).
Missing predictors are passed as `(z-scored value with NaN→0, binary missing
indicator)`.

## Pipeline

```bash
# 1) Fill gaps in the raw feature stack (GPU recommended; NN imputer only)
python scripts/harmonize_ocean_gaps.py

# 2) Build interpolation masks, native mean/std, and *_men_masked products
python scripts/build_masked_model_features.py
```

Outputs:

- `data/interim/masks/ocean_gap_stack.npy`
- `data/interim/masks/interpolation_mask_stack.npy`
- `data/interim/inputfeatures/SedTOCFeaturesnoNAN_{MAR,TOC}_{CS,DO}_men_masked/`
- `data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked/`

Each training/world row is **226-d** = 113 NN-imputed values + 113 fill indicators.
Training notebooks usually use **values only** (`MASKED_FEATURE_VARIANT=values_only`);
attention models keep the mask channels.

## Modules

| Script | Role |
|--------|------|
| `nn_feature_imputation.py` | Per-feature MLP imputer + HP search (paper) |
| `ml_feature_imputation.py` | Shared cube / native-mask helpers used by the NN imputer |
| `spatial_interpolation.py` | Polar IDW / nearest fallback, circular wave smooth |
| `mask_gap_core.py` | Gap detection + mask I/O |
| `feature_update_core.py` | Feature stack paths + r50km helpers |
| `harmonize_ocean_gaps.py` | Driver for step 1 |
| `build_interpolation_masks.py` | Fill-indicator stack |
| `recompute_feature_stats.py` | Native-ocean mean/std |
| `rebuild_harmonized_feature_products.py` | Site arrays + world chunks |
| `build_masked_model_features.py` | Orchestrates steps 2–4 |

Consumer helper for training: `src/preprocessing/masked_features.py`.

## Optional env overrides

- `FEATURES_PARTIAL_DIR` — native partial feature tree (mask diagnostics)
- `EOT20_PARTIAL_DIR` — EOT20 tide stats evidence
- `TOC_PREPROCESSED_DIR` — TOC site seeds for rebuild (default: sibling `nn-toc-update`)
