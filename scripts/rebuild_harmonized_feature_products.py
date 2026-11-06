#!/usr/bin/env python3
"""Rebuild training + world feature products from harmonized stack and masks.

Rewrites value columns and mask indicator columns from:
  - data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/
  - mask_gap_core.load_training_mask_stack()  (same as plot_masked_feature_maps)

Gap cells already hold imputed values from harmonize_ocean_gaps.py (Arctic POC,
NN/ML fills, r50km smoothed companions).  Those values are passed through
unchanged.  Only residual ocean NaNs are back-filled with the native ocean mean
(features_mean.npy) as a NaN-safety net; land stays non-finite.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "notebooks" / "SedRate"))
sys.path.insert(0, str(_SCRIPTS))

from mask_gap_core import (
    align_to_reference_grid,
    load_grid_with_coords,
    load_reference_ocean_mask,
    load_training_mask_stack,
)
from masked_features import N_ORIG_FEATURES, indicators_at_latlon

STACK = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
SEL = ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt"
MAR_MASKED_DIRS = {
    "CS": ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked",
    "DO": ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked",
}
TOC_MASKED_DIRS = {
    "CS": ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked",
    "DO": ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked",
}
TOC_PREPROCESSED = Path(
    str(__import__("os").environ.get("TOC_PREPROCESSED_DIR", "/gxfs_work/cau/sunms498/work/nn-toc-update/notebooks/TOC/preprocessed"))
)
INPUT_DIRS = [
    *MAR_MASKED_DIRS.values(),
    *TOC_MASKED_DIRS.values(),
]
WORLD_DIR = ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
CHUNK_ROWS = 6
N_COLS = 4320
N_ROWS = 2160
N_CHUNKS = N_ROWS // CHUNK_ROWS
REF_LAT = np.linspace(-90, 90, N_ROWS)
REF_LON = np.linspace(-180, 180, N_COLS)


def training_names() -> list[str]:
    names = [ln.strip() for ln in SEL.read_text().splitlines() if ln.strip()]
    # Drop the leading non-training reference column (elevation derivative).
    if names and names[0].startswith("DERIVATIVE_GL_ELEVATION"):
        return names[1:]
    return names


def load_stack_layer(fname: str) -> np.ndarray:
    arr, lat, lon = load_grid_with_coords(STACK / fname)
    return align_to_reference_grid(arr, lat, lon, REF_LAT, REF_LON)


def init_toc_masked_dirs() -> None:
    """Seed TOC masked dirs with site coordinates/labels and MAR normalization stats."""
    stats_source = MAR_MASKED_DIRS["CS"]
    for region, out_dir in TOC_MASKED_DIRS.items():
        out_dir.mkdir(parents=True, exist_ok=True)
        for src_name, dst_name in (
            (f"lats_{region}_toc.npy", "numpy_lat.npy"),
            (f"lons_{region}_toc.npy", "numpy_lon.npy"),
            (f"labels_{region}_toc.npy", "numpy_labels.npy"),
        ):
            src = TOC_PREPROCESSED / src_name
            if not src.exists():
                raise FileNotFoundError(f"Missing TOC site file: {src}")
            np.save(out_dir / dst_name, np.load(src))
        for stat_name in ("features_mean.npy", "features_std.npy"):
            shutil.copy2(stats_source / stat_name, out_dir / stat_name)
        n_sites = len(np.load(out_dir / "numpy_lat.npy"))
        print(f"  init {out_dir.name}: {n_sites} sites")


def load_native_means() -> np.ndarray:
    """Per-feature native ocean mean written by recompute_feature_stats.py.

    Filling gap cells with this value makes the z-scored gap exactly 0, i.e. the
    "unknown" neutral point, while the mask indicator channel still flags the
    cell as imputed.
    """
    for d in INPUT_DIRS:
        p = d / "features_mean.npy"
        if p.exists():
            return np.load(p).astype(np.float64)
    raise FileNotFoundError(
        "features_mean.npy not found — run recompute_feature_stats.py first"
    )


def nanfill_residual_grids(
    grids: list[np.ndarray],
    means: np.ndarray,
    ocean: np.ndarray,
) -> list[np.ndarray]:
    """Keep the NN/ML-imputed gap values; only back-fill residual ocean NaNs.

    Land cells are left non-finite (they are never used at training sites and are
    hidden on diagnostic maps).  Residual ocean NaNs receive the feature's native
    mean so z-scoring yields 0 at the neutral point while the mask channel still
    marks known gap cells.
    """
    if len(means) != len(grids):
        raise ValueError(f"means len {len(means)} != n grids {len(grids)}")
    filled: list[np.ndarray] = []
    for j, grid in enumerate(grids):
        g = np.asarray(grid, dtype=np.float64).copy()
        residual = ocean & ~np.isfinite(g)
        n_fill = int(residual.sum())
        if n_fill:
            g[residual] = means[j]
            print(f"  nan-safety [{j:3d}] {n_fill:>9,} ocean residual cells -> {means[j]:.6g}")
        filled.append(g.astype(np.float32))
    return filled


def rebuild_training(
    input_dir: Path,
    grids: list[np.ndarray],
    mask_stack: np.ndarray,
    means: np.ndarray,
) -> None:
    lats = np.load(input_dir / "numpy_lat.npy")
    lons = np.load(input_dir / "numpy_lon.npy")
    lat_grid = np.linspace(-90, 90, N_ROWS)
    lon_grid = np.linspace(-180, 180, N_COLS)
    n = len(lats)
    values = np.empty((n, N_ORIG_FEATURES), dtype=np.float32)
    for j, grid in enumerate(grids):
        r = np.array([int(np.argmin(np.abs(lat_grid - la))) for la in lats])
        c = np.array([int(np.argmin(np.abs(lon_grid - lo))) for lo in lons])
        values[:, j] = grid[r, c]
        bad = ~np.isfinite(values[:, j])
        if bad.any():
            values[bad, j] = means[j]
    indicators = indicators_at_latlon(lats, lons, mask_stack)
    out = np.concatenate([values, indicators], axis=1)
    np.save(input_dir / "numpy_features.npy", out)
    print(f"  {input_dir.name}: {out.shape}, ocean-NaN in values={np.isnan(values).sum()}")


def rebuild_world(grids: list[np.ndarray], mask_stack: np.ndarray) -> None:
    files = sorted(p for p in WORLD_DIR.iterdir() if p.suffix == ".npy")
    if len(files) != N_CHUNKS:
        raise ValueError(f"Expected {N_CHUNKS} chunks, got {len(files)}")
    for chunk_idx, path in enumerate(files):
        row0 = chunk_idx * CHUNK_ROWS
        n_rows = CHUNK_ROWS * N_COLS
        block = np.empty((n_rows, N_ORIG_FEATURES), dtype=np.float32)
        for j, grid in enumerate(grids):
            block[:, j] = grid[row0 : row0 + CHUNK_ROWS, :].reshape(-1, order="C")
        mask_block = mask_stack[row0 : row0 + CHUNK_ROWS, :, :].reshape(n_rows, N_ORIG_FEATURES)
        np.save(path, np.concatenate([block, mask_block.astype(np.float32)], axis=1))
        if chunk_idx % 60 == 0:
            print(f"  world chunk {chunk_idx + 1}/{N_CHUNKS}")
    print(f"  wrote {len(files)} world chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-world", action="store_true")
    args = parser.parse_args()

    names = training_names()
    print(f"Loading {len(names)} stack layers...")
    grids = [load_stack_layer(f) for f in names]
    ocean = load_reference_ocean_mask()
    mask_stack = load_training_mask_stack()
    print(f"Mask stack {mask_stack.shape} (interpolation_mask_stack — same as plot maps)")

    means = load_native_means()
    print("Passing through imputed stack values; ocean NaN-safety back-fill only...")
    grids = nanfill_residual_grids(grids, means, ocean)

    print("Initializing TOC masked training dirs (lat/lon/labels + stats)...")
    init_toc_masked_dirs()

    print("Rebuilding training site arrays...")
    for d in INPUT_DIRS:
        if not d.exists():
            print(f"  skip missing {d}")
            continue
        rebuild_training(d, grids, mask_stack, means)

    if not args.skip_world:
        print("Rebuilding world chunks...")
        rebuild_world(grids, mask_stack)
    print("Done.")


if __name__ == "__main__":
    main()
