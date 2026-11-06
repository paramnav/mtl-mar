# notebooks/ — Publication notebooks (data → train → models → maps)

Core set of Jupyter notebooks for the MAR manuscript: collect labels, build
features and splits, then train each machine-learning model.

Non-essential notebooks (explanations, SI extras, reproduce helpers) live in
`_archive/` and are not required for the core path.

## Workflow

```text
1. Collect labels
   collect_SR_MAR_datapoints.ipynb

2. Build site features (CS / DO) + optional edits
   create_training_features_CS.ipynb
   create_training_features_DO.ipynb
   modify_features.ipynb

3. Global ocean feature grids
   Make_Global_Features.ipynb

4. NN cross-imputation of ocean gaps (values + masks)
   nn_feature_cross_imputation.ipynb
   (or scripts under ../scripts/ — see ../scripts/README_NN_IMPUTATION.md)

5. Train / evaluate / map
   train_tree_based_models.ipynb      # RF, XGBoost, CatBoost
   train_knn.ipynb
   train_tabpfn.ipynb
   train_neural_network.ipynb         # FFNN, 113-d values
   train_multi_task_learning.ipynb    # MTL MAR+TOC, 113-d
   train_neural_network_attention.ipynb
   train_multi_task_learning_attention.ipynb

6. Operator guide (new data vs reproduce Table 2)
   run_models_on_new_data.ipynb
```

## Train notebooks (Table 2 configs)

| Notebook | Model | Notes |
|----------|--------|--------|
| `train_tree_based_models.ipynb` | RF / XGB / CatBoost | n = 700 / 1000 / 1000 |
| `train_knn.ipynb` | k-NN | k = 6 |
| `train_tabpfn.ipynb` | TabPFN | baseline |
| `train_neural_network.ipynb` | FFNN | `[128,128,128]`, values only |
| `train_multi_task_learning.ipynb` | MTL | `[128]×4`, 2 shared |
| `train_neural_network_attention.ipynb` | FFNN + attention | d256 h8 L3 |
| `train_multi_task_learning_attention.ipynb` | MTL + attention | d128 h8 L2 |

Cell outputs already contain the selected hyperparameter-search runs
(logs + scatters + maps). Refresh embeddings with:

```bash
python notebooks/embed_hp_results_into_notebooks.py
```

Helpers used by the train notebooks: `paper_train_runtime.py`,
`paper_results_display.py`.

For **new data**, follow `run_models_on_new_data.ipynb` (checklist +
`sbatch` commands). Shared train/eval code is `paper_train_runtime.py`.

To **re-install paper Table 2** maps/scatters from archived checkpoints
(same data, not new labels), use `reproduce_paper_table2.py`.

## Cluster jobs

**Preferred (one command):**

```bash
bash notebooks/run_all_models.sh            # submit all train jobs
bash notebooks/run_all_models.sh --list
bash notebooks/run_all_models.sh --gpu-only
```

Full new-data / NN-imputation / HP-search notes:
[`../docs/RETRAINING_WITH_NEW_DATA.md`](../docs/RETRAINING_WITH_NEW_DATA.md).

Individual batchfiles (same jobs):

```bash
sbatch notebooks/batchfiles/batchfile_train_trees
sbatch notebooks/batchfiles/batchfile_train_knn
sbatch notebooks/batchfiles/batchfile_train_tabpfn
sbatch notebooks/batchfiles/batchfile_train_nn
sbatch notebooks/batchfiles/batchfile_train_mtl
sbatch notebooks/batchfiles/batchfile_train_nn_attention
sbatch notebooks/batchfiles/batchfile_train_mtl_attn

# manuscript Table 2 products (checkpoints / SI figures — not new-data training)
sbatch notebooks/batchfiles/batchfile_reproduce_paper_table2
```

Monitor: `tail -f notebooks/outfiles/train_*_*.out`

## Data location

NN-imputed products are resolved in this order:

1. `mtl-mar/data/interim/.../*_men_masked`
2. fallback: sibling `nn4sedratetoc/data/interim/.../*_men_masked`

## Explainability notebooks

| Notebook | Content |
|----------|---------|
| `explain_shap_values.ipynb` | SHAP feature-importance rankings, beeswarm plots, cross-model comparison |
| `explain_information_gain.ipynb` | MC-dropout information gain, uncertainty & overconfidence maps |

These load pre-computed outputs from `nn4sedratetoc/results/` and
`nn4sedratetoc/docs_paper2/figs/`. Reproduction instructions are at the
bottom of each notebook.

## Archive

`_archive/` holds older SHAP/InfoGain/explain notebooks, reproduce scripts, and other
extras moved out of the publication path. Nothing was deleted.
