#!/usr/bin/env python3
"""Recompute features_mean.npy / features_std.npy from the global feature stack.

Stats are computed only over **native ocean cells** (ocean=True, gap_mask=False,
value finite) so that artificial zeros written by zero_invalid r50km smoothing and
large-gap features do not contaminate the normalization constants used for z-scoring.

Wave direction features use circular statistics (vector mean / circular std) because
arithmetic mean of an angle is physically meaningless and shifts the masking
transform's "neutral" z=0 value to an arbitrary degree heading.

Falls back to all finite ocean cells (equivalent to the old behaviour) when the
interpolation mask stack is unavailable.

Gap flags come from ``interpolation_mask_stack.npy`` (same masks embedded in
world cubes and shown in ``plot_masked_feature_maps``), not ``ocean_gap_stack.npy``.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))

from mask_gap_core import (
    align_to_reference_grid,
    load_grid_with_coords,
    load_reference_ocean_mask,
    load_training_mask_stack,
)

STACK = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
SEL = ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt"
INPUT_DIRS = [
    ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked",
    ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked",
]
STATS_BACKUP = ROOT / "results/feature_stats_backup"
REF_LAT = np.linspace(-90, 90, 2160)
REF_LON = np.linspace(-180, 180, 4320)

# Features whose raw values are angles in degrees — use circular statistics.
WAVE_DIRECTION_FEATURES = frozenset([
    "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd",
    "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.r50km.men.5m.grd",
])


def training_names() -> list[str]:
    names = [ln.strip() for ln in SEL.read_text().splitlines() if ln.strip()]
    # Drop the leading non-training reference column (elevation derivative).
    if names and names[0].startswith("DERIVATIVE_GL_ELEVATION"):
        return names[1:]
    return names


def backup_existing_stats(source_dir: Path) -> None:
    STATS_BACKUP.mkdir(parents=True, exist_ok=True)
    for name in ("features_mean.npy", "features_std.npy"):
        shutil.copy2(source_dir / name, STATS_BACKUP / name)
    print(f"Backed up previous stats -> {STATS_BACKUP}")


def _circular_mean_std_deg(vals: np.ndarray) -> tuple[float, float]:
    """Circular (vector) mean and circular std of angles in degrees.

    Mean:  arctan2(mean(sin), mean(cos)) — invariant to ±180° seam.
    Std:   sqrt(−2 ln R) where R = mean resultant length ∈ [0, 1].
           Converted to degrees; equals 0 for a perfectly concentrated
           distribution and grows toward ~∞ for a uniform distribution.
    """
    rad = np.deg2rad(vals.astype(np.float64))
    s = float(np.mean(np.sin(rad)))
    c = float(np.mean(np.cos(rad)))
    mean_deg = float(np.rad2deg(np.arctan2(s, c)))
    R = float(np.hypot(s, c))
    R = min(R, 1.0 - 1e-15)  # guard against exactly 1.0 → log(0)
    circ_std_deg = float(np.rad2deg(np.sqrt(-2.0 * np.log(R))))
    return mean_deg, circ_std_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print stats only; do not write files",
    )
    parser.add_argument(
        "--backup", action="store_true",
        help="Copy existing mean/std to results/feature_stats_backup/ before overwriting",
    )
    parser.add_argument(
        "--all-ocean", action="store_true",
        help="Legacy mode: use all finite ocean cells (includes zero_invalid r50km fills)",
    )
    args = parser.parse_args()

    names = training_names()

    ocean = load_reference_ocean_mask()

    gap_stack: np.ndarray | None = None
    gap_names: list[str] | None = None
    if not args.all_ocean:
        try:
            gap_stack = load_training_mask_stack()
            gap_names = names
            print(
                f"Loaded interpolation mask stack {gap_stack.shape} "
                "— computing stats on native ocean cells only"
            )
        except FileNotFoundError:
            print(
                "WARNING: interpolation mask stack not found — falling back to all finite ocean cells",
                file=sys.stderr,
            )

    mean_arr = np.empty(len(names), dtype=np.float64)
    std_arr = np.empty(len(names), dtype=np.float64)

    for i, fname in enumerate(names):
        arr, lat, lon = load_grid_with_coords(STACK / fname)
        arr = align_to_reference_grid(arr, lat, lon, REF_LAT, REF_LON)

        # Native mask: ocean + finite + not flagged as gap
        native = ocean & np.isfinite(arr)
        if gap_stack is not None and gap_names is not None and fname in gap_names:
            gap_idx = gap_names.index(fname)
            native = native & ~gap_stack[:, :, gap_idx].astype(bool)

        vals = arr[native]
        if vals.size == 0:
            # Graceful fallback: all finite ocean cells for this feature
            vals = arr[ocean & np.isfinite(arr)]
            mode_label = "fallback-all-ocean"
        elif fname in WAVE_DIRECTION_FEATURES:
            mode_label = "circular-native"
        else:
            mode_label = "native-ocean"

        if fname in WAVE_DIRECTION_FEATURES:
            m, s = _circular_mean_std_deg(vals)
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals))

        mean_arr[i] = m
        std_arr[i] = s
        print(f"[{i:3d}] {fname}: mean={m:.8g} std={s:.8g}  [{mode_label}, n={vals.size:,}]")

    if args.dry_run:
        return

    if args.backup:
        for input_dir in INPUT_DIRS:
            if input_dir.is_dir() and (input_dir / "features_mean.npy").is_file():
                backup_existing_stats(input_dir)
                break

    for input_dir in INPUT_DIRS:
        if not input_dir.is_dir():
            print(f"Skip missing: {input_dir}", file=sys.stderr)
            continue
        np.save(input_dir / "features_mean.npy", mean_arr)
        np.save(input_dir / "features_std.npy", std_arr)
        print(f"Wrote {input_dir.name}/features_mean.npy and features_std.npy")


if __name__ == "__main__":
    main()
