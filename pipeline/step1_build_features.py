#!/usr/bin/env python3
"""
Step 1 — Build training features from raw label CSVs and NetCDF feature rasters.

For each label dataset (MAR_CS, MAR_DO, TOC_CS, TOC_DO) this script:
  1. Reads the label CSV (Latitude, Longitude, target).
  2. Filters invalid rows (NaN, out-of-bounds longitude).
  3. For every selected feature raster, looks up the value at the nearest
     5 arc-minute grid cell for each data point.
  4. Saves the resulting numpy arrays to the pipeline output directory:
       numpy_features.npy, numpy_labels.npy,
       features_mean.npy, features_std.npy,
       numpy_lat.npy, numpy_lon.npy

Usage:
    python pipeline/step1_build_features.py                    # all datasets
    python pipeline/step1_build_features.py --dataset MAR_CS   # single dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# Make project importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import config as cfg


def find_nearest(value: float, vector: np.ndarray) -> int:
    return int(np.argmin(np.abs(np.float64(vector) - np.float64(value))))


def load_and_clean_labels(dataset_key: str) -> pd.DataFrame:
    """Load label CSV and keep only valid rows.

    Prefers the augmented CSV in pipeline/output/augmented_labels/ if it exists,
    otherwise falls back to the original raw label file.
    """
    spec = cfg.LABEL_FILES[dataset_key]

    aug_path = cfg.AUGMENTED_LABELS_DIR / f"{dataset_key}.csv"
    if aug_path.exists():
        csv_path = aug_path
        source = "augmented"
    else:
        csv_path = spec["csv"]
        source = "original"

    df = pd.read_csv(csv_path)
    df = df[spec["keep_cols"]]
    df = df[df["Longitude"] > -180]
    df = df.dropna(how="any")
    print(f"[{dataset_key}] Loaded {len(df)} valid label rows from {csv_path.name} ({source})")
    return df


def get_selected_feature_files() -> list[str]:
    """Return the sorted list of feature files that pass the selection filter."""
    # selectedfeatures_v_men.txt is a plain line list (no CSV header)
    raw = cfg.FEATURE_SELECTION_FILE.read_text().splitlines()
    feature_selection = [ln.strip() for ln in raw if ln.strip()]
    # Drop leading elevation-derivative reference if present (training uses 113)
    if feature_selection and feature_selection[0].startswith("DERIVATIVE_GL_ELEVATION"):
        feature_selection = feature_selection[1:]

    all_files = sorted(f.name for f in cfg.RAW_FEATURES_DIR.iterdir() if f.is_file())
    selected = [f for f in all_files if f in feature_selection]
    # Preserve selection order when possible
    selected = [f for f in feature_selection if f in set(all_files)]
    print(f"Selected {len(selected)} / {len(all_files)} feature files")
    return selected


def extract_features(df: pd.DataFrame, feature_files: list[str]) -> tuple:
    """Extract feature values at label locations from NetCDF rasters.

    Returns (features, X_mean, X_std, lats, lons).
    """
    n_samples = len(df)
    n_features = len(feature_files)

    features = np.full((n_samples, n_features), np.nan)
    X_mean = np.empty(n_features)
    X_std = np.empty(n_features)

    # Build lat/lon index arrays once (5 arc-min grid)
    grid_lats = np.linspace(-90, 90, cfg.N_LAT)
    grid_lons = np.linspace(-180, 180, cfg.N_LON)

    # Pre-compute nearest indices for all data points
    lat_idx = np.array([find_nearest(df.iloc[i, 0], grid_lats) for i in range(n_samples)])
    lon_idx = np.array([find_nearest(df.iloc[i, 1], grid_lons) for i in range(n_samples)])

    corrupted = 0
    for fi, fname in enumerate(feature_files):
        fpath = cfg.RAW_FEATURES_DIR / fname
        try:
            ds = xr.open_dataset(fpath, engine="netcdf4")
            data = ds["data"].values
        except Exception:
            X_mean[fi] = np.nan
            X_std[fi] = np.nan
            corrupted += 1
            continue

        X_mean[fi] = np.nanmean(data)
        X_std[fi] = np.nanstd(data)

        for si in range(n_samples):
            features[si, fi] = data[lat_idx[si], lon_idx[si]]

        if (fi + 1) % 20 == 0 or fi == n_features - 1:
            print(f"  feature {fi+1}/{n_features}: {fname}")

    if corrupted:
        print(f"  WARNING: {corrupted} corrupted feature files skipped")

    lats = np.array(df.iloc[:, 0])
    lons = np.array(df.iloc[:, 1])
    return features, X_mean, X_std, lats, lons


def save_features(dataset_key: str, features, labels, X_mean, X_std, lats, lons):
    """Save extracted features to the pipeline output area."""
    out_dir = cfg.PIPELINE_OUTPUT_DIR / "features" / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "numpy_features.npy", features)
    np.save(out_dir / "numpy_labels.npy", labels)
    np.save(out_dir / "features_mean.npy", X_mean)
    np.save(out_dir / "features_std.npy", X_std)
    np.save(out_dir / "numpy_lat.npy", lats)
    np.save(out_dir / "numpy_lon.npy", lons)
    print(f"[{dataset_key}] Saved features → {out_dir}")


def build_dataset(dataset_key: str):
    spec = cfg.LABEL_FILES[dataset_key]
    df = load_and_clean_labels(dataset_key)
    feature_files = get_selected_feature_files()
    features, X_mean, X_std, lats, lons = extract_features(df, feature_files)
    labels = df[spec["target_col"]].values
    save_features(dataset_key, features, labels, X_mean, X_std, lats, lons)


def main():
    parser = argparse.ArgumentParser(description="Step 1: Build training features")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Build features for a single dataset (MAR_CS, MAR_DO, TOC_CS, TOC_DO)")
    args = parser.parse_args()

    if args.dataset:
        if args.dataset not in cfg.LABEL_FILES:
            print(f"ERROR: unknown dataset '{args.dataset}'. Choose from {list(cfg.LABEL_FILES.keys())}")
            sys.exit(1)
        build_dataset(args.dataset)
    else:
        for key in cfg.LABEL_FILES:
            build_dataset(key)


if __name__ == "__main__":
    main()
