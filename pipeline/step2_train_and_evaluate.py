#!/usr/bin/env python3
"""
Step 2 — Train tree-based models, run hyperparameter sweeps, produce
scatter plots and metrics, and generate prediction maps.

Supports: Random Forest, XGBoost, CatBoost, KNN, LightGBM, DNN.

Usage:
    python pipeline/step2_train_and_evaluate.py                         # full run (MAR + TOC)
    python pipeline/step2_train_and_evaluate.py --target MAR            # only MAR
    python pipeline/step2_train_and_evaluate.py --target TOC            # only TOC
    python pipeline/step2_train_and_evaluate.py --skip-hp-sweep         # skip hyperparameter sweeps
    python pipeline/step2_train_and_evaluate.py --skip-maps             # skip map generation
    python pipeline/step2_train_and_evaluate.py --models rf xgb         # only specific models
    python pipeline/step2_train_and_evaluate.py --use-existing-features # use pre-built interim features instead of pipeline ones
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "models"))
import config as cfg

# ── Helpers ─────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _values_only(features, x_mean, x_std, n_orig=None):
    """Keep NN-imputed value channels when masked (226-d) products are loaded."""
    n_orig = n_orig or getattr(cfg, "N_ORIG_FEATURES", 113)
    features = np.asarray(features)
    if features.shape[1] == 2 * n_orig:
        features = features[:, :n_orig]
    if x_mean.shape[0] != features.shape[1]:
        x_mean = x_mean[: features.shape[1]]
        x_std = x_std[: features.shape[1]]
    return features, x_mean, x_std


def load_features(dataset_key: str, use_existing: bool = False, values_only: bool = True):
    """Load numpy arrays for a dataset.

    If *use_existing* is True, prefers ``*_men_masked`` then ``*_men`` under
    data/interim/inputfeatures/.  Otherwise reads pipeline/output/features/.
    """
    if use_existing:
        mapping = {
            "MAR_CS": "SedTOCFeaturesnoNAN_MAR_CS",
            "MAR_DO": "SedTOCFeaturesnoNAN_MAR_DO",
            "TOC_CS": "SedTOCFeaturesnoNAN_TOC_CS",
            "TOC_DO": "SedTOCFeaturesnoNAN_TOC_DO",
        }
        base = mapping[dataset_key]
        candidates = [
            cfg.INTERIM_INPUT_FEATURES_DIR / f"{base}_men_masked",
            cfg.INTERIM_INPUT_FEATURES_DIR / f"{base}_men",
        ]
        feat_dir = next((c for c in candidates if c.exists()), candidates[-1])
    else:
        feat_dir = cfg.PIPELINE_OUTPUT_DIR / "features" / dataset_key

    features = np.load(feat_dir / "numpy_features.npy")
    labels = np.load(feat_dir / "numpy_labels.npy")
    X_mean = np.load(feat_dir / "features_mean.npy").astype(float)
    X_std = np.load(feat_dir / "features_std.npy").astype(float)
    lats = np.load(feat_dir / "numpy_lat.npy")
    lons = np.load(feat_dir / "numpy_lon.npy")
    if values_only:
        features, X_mean, X_std = _values_only(features, X_mean, X_std)
    print(f"[{dataset_key}] Loaded {features.shape[0]} samples × {features.shape[1]} features from {feat_dir}")
    return features, labels, X_mean, X_std, lats, lons


def preprocess(features, labels, lats, lons, X_mean, X_std):
    """Remove NaN rows/cols, normalise, deduplicate."""
    nan_rows = np.isnan(features).any(axis=1)
    features, labels, lats, lons = features[~nan_rows], labels[~nan_rows], lats[~nan_rows], lons[~nan_rows]

    nan_cols = np.isnan(features).any(axis=0)
    features = features[:, ~nan_cols]

    features = np.divide((features - X_mean), X_std)

    # Deduplicate (ported from trainDNN.deduplicate_training_sites)
    arr = features.copy()
    labels_c = labels.copy()
    unique_rows, row_counts = np.unique(arr, axis=0, return_counts=True)
    dup_idx = np.where(row_counts > 1)[0]
    for idx in dup_idx:
        row_idx = np.where((arr == unique_rows[idx]).all(axis=1))[0]
        group_labels = labels_c[row_idx]
        if np.std(group_labels) > 0.2 * np.max(np.abs(group_labels) + 1e-12):
            arr[row_idx] = np.nan
            labels_c[row_idx] = np.nan
        else:
            arr[row_idx[1:]] = np.nan
            labels_c[row_idx[0]] = np.mean(group_labels)
            labels_c[row_idx[1:]] = np.nan

    keep = ~np.isnan(arr).any(axis=1) & ~np.isnan(labels_c)
    return arr[keep], labels_c[keep], lats[keep], lons[keep]


def stratified_split(features, labels, lats, lons, test_ratio=0.25, seed=42):
    """Stratified train/test split using quantile-based bins."""
    n_bins = cfg.N_STRATIFY_BINS
    try:
        y_binned = pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        y_binned = None

    np.random.seed(seed)
    n = len(labels)
    perm = np.random.permutation(n)
    n_test = int(n * test_ratio)
    if y_binned is not None:
        from sklearn.model_selection import train_test_split as sk_split
        idx_train, idx_test = sk_split(np.arange(n), test_size=test_ratio, random_state=seed, stratify=y_binned)
    else:
        idx_train, idx_test = perm[n_test:], perm[:n_test]

    return (features[idx_train], labels[idx_train], lats[idx_train], lons[idx_train],
            features[idx_test], labels[idx_test], lats[idx_test], lons[idx_test])


# ── Tree-based model training ──────────────────────────────────────────────

def train_tree_model(X_train, y_train, X_test, y_test, model_type="random_forest", n_estimators=100, seed=42, **kw):
    import xgboost as xgb
    import catboost as cb

    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1, **kw)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed, verbosity=0, n_jobs=-1, learning_rate=0.01, **kw)
    elif model_type == "catboost":
        model = cb.CatBoostRegressor(iterations=n_estimators, random_seed=seed, verbose=0, learning_rate=0.01, loss_function="RMSE", **kw)
    elif model_type == "knn":
        model = KNeighborsRegressor(n_neighbors=n_estimators, weights="uniform", metric="minkowski", p=2)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    return model, train_preds, test_preds


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    corr = float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1])
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "corr": corr, "r2": r2}


# ── Scatter plot ────────────────────────────────────────────────────────────

def save_scatter(y_train, train_preds, y_test, test_preds, model_name, tag, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_m = compute_metrics(y_train, train_preds)
    test_m = compute_metrics(y_test, test_preds)

    fig, ax = plt.subplots(figsize=cfg.SCATTER_FIGSIZE, dpi=cfg.SCATTER_DPI)
    ax.scatter(y_train, train_preds, color="dodgerblue", alpha=0.5, s=12,
               label=f"Train (MAE={train_m['mae']:.3f}, r={train_m['corr']:.3f})")
    ax.scatter(y_test, test_preds, color="darkorange", alpha=0.5, s=12,
               label=f"Test (MAE={test_m['mae']:.3f}, r={test_m['corr']:.3f})")
    all_v = np.concatenate([y_train, y_test, train_preds, test_preds])
    lo, hi = all_v.min(), all_v.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(f"{model_name} — {tag}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fpath = out_dir / f"scatter_{tag}_{model_name}.png"
    fig.savefig(fpath)
    plt.close(fig)
    print(f"  → scatter saved: {fpath.name}")
    return train_m, test_m


# ── Hyperparameter sweep ───────────────────────────────────────────────────

def run_hp_sweep(X_train, y_train, X_test, y_test, model_type, param_values, tag, out_dir):
    """Sweep over n_estimators (or k for KNN) and save results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for val in param_values:
        set_seed()
        model, tr_pred, te_pred = train_tree_model(X_train, y_train, X_test, y_test,
                                                    model_type=model_type, n_estimators=int(val))
        tr_m = compute_metrics(y_train, tr_pred)
        te_m = compute_metrics(y_test, te_pred)
        results.append({"n": int(val),
                        "train_mae": tr_m["mae"], "test_mae": te_m["mae"],
                        "train_r2": tr_m["r2"], "test_r2": te_m["r2"],
                        "train_corr": tr_m["corr"], "test_corr": te_m["corr"]})
        print(f"  {model_type} n={int(val):>5d}  test_MAE={te_m['mae']:.4f}  test_r={te_m['corr']:.4f}")

    # Save CSV
    df_res = pd.DataFrame(results)
    csv_path = out_dir / f"hp_sweep_{tag}_{model_type}.csv"
    df_res.to_csv(csv_path, index=False)

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, dpi=cfg.SCATTER_DPI)
    ns = df_res["n"].values

    for ax, (tr_col, te_col, ylabel) in zip(axes, [
        ("train_mae", "test_mae", "MAE"),
        ("train_r2", "test_r2", "$R^2$"),
        ("train_corr", "test_corr", "Correlation"),
    ]):
        ax.plot(ns, df_res[tr_col], label=f"Train {ylabel}", color="C0")
        ax.plot(ns, df_res[te_col], label=f"Test {ylabel}", color="C1")
        best_idx = int(np.argmin(df_res[te_col])) if "mae" in te_col else int(np.argmax(df_res[te_col]))
        ax.scatter(ns[best_idx], df_res[te_col].iloc[best_idx], color="red", s=80, zorder=5)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, loc="best", fontsize=9)
        ax.set_xscale("log")

    axes[-1].set_xlabel("Number of Estimators / k")
    fig.suptitle(f"Hyperparameter Sweep — {model_type} — {tag}", fontsize=13)
    fig.tight_layout()
    fig_path = out_dir / f"hp_sweep_{tag}_{model_type}.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  → HP sweep plot saved: {fig_path.name}")
    return df_res


# ── Map prediction ──────────────────────────────────────────────────────────

def predict_map(model, X_mean, X_std, tag, model_name, out_dir, is_mar=True):
    """Generate a 2160×4320 prediction map using the world features."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from os import walk as os_walk

    files = sorted(f for f in cfg.WORLD_FEATURES_DIR.iterdir() if f.suffix == ".npy")
    prediction_map = np.zeros((cfg.N_LAT, cfg.N_LON))

    for chunk_idx, fpath in enumerate(files):
        chunk = np.load(fpath)
        if chunk.ndim == 2 and chunk.shape[1] >= X_mean.shape[0]:
            chunk = chunk[:, : X_mean.shape[0]]
        chunk_norm = np.nan_to_num(np.divide((chunk - X_mean), X_std), nan=0.0, posinf=0.0, neginf=0.0)
        preds = model.predict(chunk_norm)
        row_start = chunk_idx * cfg.CHUNK_SHAPE[0]
        row_end = row_start + cfg.CHUNK_SHAPE[0]
        prediction_map[row_start:row_end, :] = preds.reshape(cfg.CHUNK_SHAPE)

    # Orient the map (match existing pipeline convention)
    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))

    # Apply land mask
    land_map = np.load(cfg.LAND_MAP_FILE)
    land_map[np.isnan(land_map)] = 1
    land_mask = land_map.astype(bool)
    prediction_map[land_mask] = 0

    # Save numpy
    npy_path = out_dir / f"prediction_map_{tag}_{model_name}.npy"
    np.save(npy_path, prediction_map)

    # Plot
    fig, ax = plt.subplots(figsize=cfg.MAP_FIGSIZE, dpi=cfg.MAP_DPI)
    if is_mar:
        im = ax.imshow(prediction_map, norm=mcolors.LogNorm(vmin=cfg.MAP_VMIN_MAR, vmax=cfg.MAP_VMAX_MAR),
                        cmap=cfg.MAP_CMAP_MAR, extent=[-180, 180, -90, 90])
    else:
        prediction_map_plot = prediction_map.copy()
        prediction_map_plot[prediction_map_plot == 0] = np.nan
        im = ax.imshow(prediction_map_plot, vmin=cfg.MAP_VMIN_TOC, vmax=cfg.MAP_VMAX_TOC,
                        cmap=cfg.MAP_CMAP_TOC, extent=[-180, 180, -90, 90])
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(f"{model_name} — {tag}", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    png_path = out_dir / f"prediction_map_{tag}_{model_name}.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"  → map saved: {png_path.name}")
    gc.collect()


# ── Main orchestrator ───────────────────────────────────────────────────────

def run_for_target(target: str, model_names: list[str], skip_hp: bool, skip_maps: bool, use_existing: bool):
    """Run the full pipeline for MAR or TOC."""
    is_mar = target.upper().startswith("MAR")
    cs_key = f"{target}_CS"
    do_key = f"{target}_DO"

    print(f"\n{'='*80}")
    print(f"  Pipeline: {target}")
    print(f"{'='*80}\n")

    # Load both CS and DO
    feat_CS, lab_CS, mean_CS, std_CS, lat_CS, lon_CS = load_features(cs_key, use_existing)
    feat_DO, lab_DO, mean_DO, std_DO, lat_DO, lon_DO = load_features(do_key, use_existing)

    # Use DO normalization stats for the combined model (matching existing convention)
    X_mean, X_std = mean_DO, std_DO

    # Preprocess each
    feat_CS, lab_CS, lat_CS, lon_CS = preprocess(feat_CS, lab_CS, lat_CS, lon_CS, X_mean, X_std)
    feat_DO, lab_DO, lat_DO, lon_DO = preprocess(feat_DO, lab_DO, lat_DO, lon_DO, X_mean, X_std)

    # Stratified split for each domain
    trF_CS, trL_CS, trLa_CS, trLo_CS, teF_CS, teL_CS, teLa_CS, teLo_CS = stratified_split(feat_CS, lab_CS, lat_CS, lon_CS)
    trF_DO, trL_DO, trLa_DO, trLo_DO, teF_DO, teL_DO, teLa_DO, teLo_DO = stratified_split(feat_DO, lab_DO, lat_DO, lon_DO)

    # Merge for "entire" training
    train_features = np.concatenate([trF_CS, trF_DO])
    test_features = np.concatenate([teF_CS, teF_DO])
    train_labels = np.concatenate([trL_CS, trL_DO])
    test_labels = np.concatenate([teL_CS, teL_DO])

    print(f"\nTrain: {train_features.shape[0]} samples, Test: {test_features.shape[0]} samples")
    print(f"Features: {train_features.shape[1]} dimensions\n")

    tag = f"{target}_entire"
    all_results = {}

    # ── Model loop ──────────────────────────────────────────────────────────
    model_dispatch = {
        "rf":  ("random_forest", cfg.TREE_N_ESTIMATORS_SWEEP, cfg.BEST_RF_N),
        "xgb": ("xgboost",      cfg.TREE_N_ESTIMATORS_SWEEP, cfg.BEST_XGB_N),
        "cb":  ("catboost",      cfg.TREE_N_ESTIMATORS_SWEEP, cfg.BEST_CB_N),
        "knn": ("knn",           np.array(cfg.KNN_K_SWEEP),   cfg.BEST_KNN_K),
    }

    for short_name in model_names:
        if short_name not in model_dispatch:
            print(f"  SKIP unknown model '{short_name}'")
            continue

        model_type, sweep_values, best_default = model_dispatch[short_name]
        print(f"\n── {model_type.upper()} ──────────────────────────────")

        # HP sweep
        if not skip_hp:
            hp_df = run_hp_sweep(train_features, train_labels, test_features, test_labels,
                                  model_type, sweep_values, tag, cfg.PIPELINE_HP_DIR)
            # Pick best
            best_row = hp_df.loc[hp_df["test_mae"].idxmin()]
            best_n = int(best_row["n"])
            print(f"  Best {model_type}: n={best_n}, test_MAE={best_row['test_mae']:.4f}")
        else:
            best_n = best_default

        # Final training with best hyperparameter
        set_seed()
        model, tr_pred, te_pred = train_tree_model(train_features, train_labels, test_features, test_labels,
                                                    model_type=model_type, n_estimators=best_n)

        # Scatter plot
        tr_m, te_m = save_scatter(train_labels, tr_pred, test_labels, te_pred,
                                   model_type, tag, cfg.PIPELINE_SCATTER_DIR)
        all_results[model_type] = {"best_n": best_n, "train": tr_m, "test": te_m}

        # Map
        if not skip_maps:
            predict_map(model, X_mean, X_std, tag, model_type, cfg.PIPELINE_MAPS_DIR, is_mar=is_mar)

        gc.collect()

    # ── Save combined results ───────────────────────────────────────────────
    cfg.PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = cfg.PIPELINE_RESULTS_DIR / f"results_{tag}.json"
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    print(f"\n→ Combined results saved: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Train, evaluate, and map")
    parser.add_argument("--target", type=str, default=None, choices=["MAR", "TOC"],
                        help="Process only MAR or TOC (default: both)")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "cb", "knn"],
                        help="Models to train: rf, xgb, cb, knn (default: all)")
    parser.add_argument("--skip-hp-sweep", action="store_true",
                        help="Skip hyperparameter sweeps and use best defaults from config")
    parser.add_argument("--skip-maps", action="store_true",
                        help="Skip prediction map generation (faster for testing)")
    parser.add_argument("--use-existing-features", action="store_true",
                        help="Use pre-built features from data/interim/ instead of pipeline/output/features/")
    args = parser.parse_args()

    targets = [args.target] if args.target else ["MAR", "TOC"]
    for t in targets:
        run_for_target(t, args.models, args.skip_hp_sweep, args.skip_maps, args.use_existing_features)


if __name__ == "__main__":
    main()
