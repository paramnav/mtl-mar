#!/usr/bin/env python3
"""
Utility to add new data points to the label CSVs and rebuild features.

Adds rows to the raw label CSV files without touching the originals — instead
it creates an *augmented* copy in  pipeline/output/augmented_labels/  and then
re-runs step 1 against the augmented file.

Supports three input modes:
  1. From a CSV file with the same format (Latitude, Longitude, target).
  2. From command-line arguments for a single point.
  3. Interactive prompt.

Examples:
    # Append rows from a CSV
    python pipeline/add_data.py --dataset MAR_CS --csv new_points.csv

    # Add a single point
    python pipeline/add_data.py --dataset MAR_CS --lat -38.5 --lon 178.2 --value 120.5

    # Interactive mode
    python pipeline/add_data.py --dataset MAR_CS --interactive

    # After adding, rebuild features and retrain:
    python pipeline/run_pipeline.py --skip-tree --skip-dnn   # just rebuild features
    python pipeline/run_pipeline.py                          # full pipeline
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
import config as cfg

AUGMENTED_DIR = cfg.PIPELINE_OUTPUT_DIR / "augmented_labels"


def get_augmented_csv(dataset_key: str) -> Path:
    """Return path to the augmented CSV.  Create from original if needed."""
    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    spec = cfg.LABEL_FILES[dataset_key]
    aug_path = AUGMENTED_DIR / f"{dataset_key}.csv"

    if not aug_path.exists():
        shutil.copy2(spec["csv"], aug_path)
        print(f"Created augmented copy: {aug_path}")

    return aug_path


def add_from_csv(dataset_key: str, csv_path: str):
    """Append rows from *csv_path* into the augmented label CSV."""
    spec = cfg.LABEL_FILES[dataset_key]
    aug_path = get_augmented_csv(dataset_key)

    existing = pd.read_csv(aug_path)
    new = pd.read_csv(csv_path)

    # Validate required columns
    target_col = spec["target_col"]
    for col in ["Latitude", "Longitude", target_col]:
        if col not in new.columns:
            print(f"ERROR: column '{col}' not found in {csv_path}")
            print(f"  Expected columns: Latitude, Longitude, {target_col}")
            sys.exit(1)

    n_before = len(existing)
    combined = pd.concat([existing, new[spec["keep_cols"]]], ignore_index=True)
    combined.to_csv(aug_path, index=False)
    n_after = len(combined)
    print(f"[{dataset_key}] Added {n_after - n_before} new points → {n_after} total in {aug_path.name}")


def add_single_point(dataset_key: str, lat: float, lon: float, value: float):
    """Add a single data point."""
    spec = cfg.LABEL_FILES[dataset_key]
    aug_path = get_augmented_csv(dataset_key)

    existing = pd.read_csv(aug_path)
    new_row = pd.DataFrame([{
        "Latitude": lat,
        "Longitude": lon,
        spec["target_col"]: value,
    }])
    combined = pd.concat([existing, new_row[spec["keep_cols"]]], ignore_index=True)
    combined.to_csv(aug_path, index=False)
    print(f"[{dataset_key}] Added point (lat={lat}, lon={lon}, {spec['target_col']}={value})")
    print(f"  Total: {len(combined)} rows in {aug_path.name}")


def add_interactive(dataset_key: str):
    """Interactively add points one by one."""
    spec = cfg.LABEL_FILES[dataset_key]
    target_col = spec["target_col"]
    print(f"\nAdding points to {dataset_key}. Enter 'done' to finish.\n")

    while True:
        try:
            line = input(f"  Latitude Longitude {target_col} (or 'done'): ").strip()
        except EOFError:
            break
        if line.lower() in ("done", "quit", "exit", "q"):
            break
        parts = line.split()
        if len(parts) != 3:
            print("  → Expected 3 space-separated values. Try again.")
            continue
        try:
            lat, lon, val = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            print("  → Could not parse numbers. Try again.")
            continue
        add_single_point(dataset_key, lat, lon, val)

    print("\nDone adding points.")
    print(f"To rebuild features: python pipeline/step1_build_features.py --dataset {dataset_key}")
    print(f"To run full pipeline: python pipeline/run_pipeline.py")


def show_summary(dataset_key: str):
    """Print current data summary."""
    spec = cfg.LABEL_FILES[dataset_key]
    orig = pd.read_csv(spec["csv"])
    orig_clean = orig[spec["keep_cols"]].dropna()

    aug_path = AUGMENTED_DIR / f"{dataset_key}.csv"
    if aug_path.exists():
        aug = pd.read_csv(aug_path)
        aug_clean = aug[spec["keep_cols"]].dropna()
        n_new = len(aug_clean) - len(orig_clean)
        print(f"\n[{dataset_key}] Original: {len(orig_clean)} | Augmented: {len(aug_clean)} | New: {n_new}")
        print(f"  Lat range:   {aug_clean['Latitude'].min():.2f} to {aug_clean['Latitude'].max():.2f}")
        print(f"  Lon range:   {aug_clean['Longitude'].min():.2f} to {aug_clean['Longitude'].max():.2f}")
        tc = spec["target_col"]
        print(f"  {tc} range: {aug_clean[tc].min():.4f} to {aug_clean[tc].max():.4f}")
    else:
        print(f"\n[{dataset_key}] Original: {len(orig_clean)} (no augmented data yet)")


def main():
    parser = argparse.ArgumentParser(description="Add new data points to the pipeline")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(cfg.LABEL_FILES.keys()),
                        help="Target dataset: MAR_CS, MAR_DO, TOC_CS, TOC_DO")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV file with new data points")
    parser.add_argument("--lat", type=float, default=None)
    parser.add_argument("--lon", type=float, default=None)
    parser.add_argument("--value", type=float, default=None)
    parser.add_argument("--interactive", action="store_true",
                        help="Add points interactively")
    parser.add_argument("--summary", action="store_true",
                        help="Show current data summary")
    args = parser.parse_args()

    if args.summary:
        show_summary(args.dataset)
        return

    if args.csv:
        add_from_csv(args.dataset, args.csv)
    elif args.lat is not None and args.lon is not None and args.value is not None:
        add_single_point(args.dataset, args.lat, args.lon, args.value)
    elif args.interactive:
        add_interactive(args.dataset)
    else:
        print("ERROR: provide --csv, (--lat --lon --value), or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
