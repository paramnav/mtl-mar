#!/usr/bin/env python3
"""Rebuild masked MAR training features and global world cubes for model runs.

Run after updating the feature stack (e.g. harmonize_ocean_gaps.py, Arctic POC
enrichment, r50km rebuild).  All three steps share the same mask definition
(``interpolation_mask_stack.npy``) that ``plot_masked_feature_maps`` displays.

Pipeline:
  1. build_interpolation_masks.py   — refresh gap indicators from current stack
  2. recompute_feature_stats.py     — native-ocean mean/std (excludes gap cells)
  3. rebuild_harmonized_feature_products.py — site arrays + 360 world chunks

Do not use ``--skip-masks`` after stack or gap-filling changes — stale masks
make training stats and diagnostic maps disagree.

Prerequisites (one-time / already done):
  - Label CSVs + numpy_lat/lon/numpy_labels in each INPUT_DIR
  - Feature stack at data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated/
  - 360 world chunk files already exist (this script overwrites them in place)

Usage:
  python3 scripts/build_masked_model_features.py
  python3 scripts/build_masked_model_features.py --skip-masks
  python3 scripts/build_masked_model_features.py --skip-world
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_step(label: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def verify_outputs() -> None:
    import numpy as np

    sys.path.insert(0, str(SCRIPTS))
    from mask_gap_core import (
        align_to_reference_grid,
        load_grid_with_coords,
        load_reference_ocean_mask,
        load_training_mask_stack,
        training_feature_names,
    )

    ref_lat = np.linspace(-90, 90, 2160)
    ref_lon = np.linspace(-180, 180, 4320)

    cs = ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked"
    do = ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked"
    toc_cs = ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked"
    toc_do = ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked"
    world = ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
    stack = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
    ocean = load_reference_ocean_mask()
    means = np.load(cs / "features_mean.npy")

    for d, name in (
        (cs, "MAR CS"),
        (do, "MAR DO"),
        (toc_cs, "TOC CS"),
        (toc_do, "TOC DO"),
    ):
        feats = np.load(d / "numpy_features.npy")
        mean = np.load(d / "features_mean.npy")
        std = np.load(d / "features_std.npy")
        n_feat = len(mean)
        n_nan = int(np.isnan(feats[:, :n_feat]).sum())
        print(f"  {name}: features {feats.shape}, mean/std len {len(mean)}/{len(std)}, value NaN={n_nan}")

    chunks = sorted(world.glob("Sed_world_feature_*.npy"))
    if chunks:
        sample = np.load(chunks[0], mmap_mode="r")
        print(f"  world: {len(chunks)} chunks, cols={sample.shape[1]} (expect 2x n_features)")

    # Spot-check: world value/mask channels match stack + interpolation masks.
    names = training_feature_names()
    mask_stack = load_training_mask_stack()
    check_idx = [names.index(n) for n in (
        "SS_POC_LOG_MOL_M3-1_VIIRS_SNPP_MISSION_MEANx.5m.grd",
        "SS_POC_LOG_MOL_M3-1_VIIRS_SNPP_MISSION_MEANx.r50km.men.5m.grd",
    ) if n in names]
    chunk0 = np.load(chunks[0])
    n_feat = len(names)
    for j in check_idx:
        arr, lat, lon = load_grid_with_coords(stack / names[j])
        stack_val = align_to_reference_grid(arr, lat, lon, ref_lat, ref_lon)
        filled = stack_val[:6].astype(np.float64).copy()
        resid = ocean[:6] & ~np.isfinite(filled)
        if resid.any():
            filled[resid] = means[j]
        world_val = chunk0[:, j].reshape(6, 4320)
        world_msk = chunk0[:, n_feat + j].reshape(6, 4320)
        val_ok = np.allclose(filled, world_val, equal_nan=True, rtol=1e-5, atol=1e-4)
        msk_ok = np.array_equal(mask_stack[:6, :, j], world_msk)
        print(f"  align {names[j]}: values={val_ok} masks={msk_ok}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-masks", action="store_true", help="Skip interpolation mask rebuild")
    parser.add_argument("--skip-world", action="store_true", help="Rebuild training only")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_masks:
        run_step("Step 1/3: interpolation masks", [py, str(SCRIPTS / "build_interpolation_masks.py")])
    else:
        print(
            "\nWARNING: --skip-masks — using existing interpolation_mask_stack.npy; "
            "ensure it matches the current feature stack.",
            file=sys.stderr,
        )

    run_step("Step 2/3: feature mean/std", [py, str(SCRIPTS / "recompute_feature_stats.py"), "--backup"])

    cmd = [py, str(SCRIPTS / "rebuild_harmonized_feature_products.py")]
    if args.skip_world:
        cmd.append("--skip-world")
    run_step("Step 3/3: training + world features", cmd)

    print(f"\n{'=' * 72}\nVerification\n{'=' * 72}")
    verify_outputs()
    print("\nDone. Training dirs ready for masked MAR models; world cubes ready for prediction.")


if __name__ == "__main__":
    main()
