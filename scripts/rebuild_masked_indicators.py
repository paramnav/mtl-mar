#!/usr/bin/env python3
"""Refresh mask indicator columns in stored masked feature products.

Raw predictor values (cols 0-110) are kept; indicator columns (cols 111-221) are
rebuilt from ``load_filtered_mask_stack()`` after mask-stack updates (e.g. Garlan).

Updates:
  - data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_{CS,DO}_men_masked/
  - data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked/

Optionally clears Hayes task caches so they are rebuilt with the new masks.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "SedRate"))

from masked_features import N_ORIG_FEATURES, indicators_at_latlon, load_filtered_mask_stack

N_MASK = N_ORIG_FEATURES
CHUNK_ROWS = 6
N_COLS = 4320
N_ROWS = 2160
N_CHUNKS = N_ROWS // CHUNK_ROWS

INPUT_DIRS = [
    ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked",
    ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked",
]
WORLD_DIR = ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
HAYES_CACHE = ROOT / "data/interim/hayes_tasks"


def rebuild_training_site_features(input_dir: Path, mask_stack: np.ndarray) -> None:
    features = np.load(input_dir / "numpy_features.npy")
    lats = np.load(input_dir / "numpy_lat.npy")
    lons = np.load(input_dir / "numpy_lon.npy")
    if features.shape[1] != 2 * N_ORIG_FEATURES:
        raise ValueError(f"{input_dir}: expected {2 * N_ORIG_FEATURES} columns, got {features.shape[1]}")

    values = features[:, :N_ORIG_FEATURES]
    indicators = indicators_at_latlon(lats, lons, mask_stack)
    updated = np.concatenate([values, indicators], axis=1)
    np.save(input_dir / "numpy_features.npy", updated)
    print(f"Updated {input_dir.name}: {updated.shape[0]} sites, indicators mean={indicators.mean():.4f}")


def rebuild_world_chunks(world_dir: Path, mask_stack: np.ndarray) -> None:
    files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
    if len(files) != N_CHUNKS:
        raise ValueError(f"Expected {N_CHUNKS} world chunks in {world_dir}, found {len(files)}")

    for chunk_idx, path in enumerate(files):
        features = np.load(path)
        if features.shape[1] != 2 * N_ORIG_FEATURES:
            raise ValueError(f"{path.name}: expected {2 * N_ORIG_FEATURES} columns, got {features.shape[1]}")
        values = features[:, :N_ORIG_FEATURES]
        row0 = chunk_idx * CHUNK_ROWS
        mask_block = mask_stack[row0 : row0 + CHUNK_ROWS, :, :]
        indicators = mask_block.reshape(CHUNK_ROWS * N_COLS, N_MASK)
        updated = np.concatenate([values, indicators], axis=1)
        np.save(path, updated)
        if (chunk_idx + 1) % 60 == 0 or chunk_idx == 0:
            print(f"  world chunk {chunk_idx + 1}/{N_CHUNKS}")

    print(f"Updated {len(files)} world chunks in {world_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clear-hayes-cache",
        action="store_true",
        help="Remove data/interim/hayes_tasks so Hayes datasets rebuild on next run",
    )
    args = parser.parse_args()

    print("Loading filtered mask stack...")
    mask_stack = load_filtered_mask_stack()
    print(f"Mask stack shape: {mask_stack.shape}")

    names = [
        ln.strip()
        for ln in (ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt")
        .read_text()
        .splitlines()
        if ln.strip()
    ]
    training_names = names[1:] if len(names) == N_ORIG_FEATURES + 1 else names
    litho_idx = training_names.index("litho_maps_type1_.nc")
    litho_mask_frac = float(mask_stack[:, :, litho_idx].mean())
    print(f"litho_maps_type1_.nc mask fraction (all cells): {litho_mask_frac:.4f}")

    for input_dir in INPUT_DIRS:
        rebuild_training_site_features(input_dir, mask_stack)

    rebuild_world_chunks(WORLD_DIR, mask_stack)

    if args.clear_hayes_cache and HAYES_CACHE.exists():
        shutil.rmtree(HAYES_CACHE)
        print(f"Cleared Hayes cache: {HAYES_CACHE}")

    print("Done.")


if __name__ == "__main__":
    main()
