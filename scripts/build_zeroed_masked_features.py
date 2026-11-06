#!/usr/bin/env python3
"""Build a no-imputation ablation of the masked feature products.

Current ``*_men_masked`` products keep NN/ML-imputed values in gap cells and
attach a binary indicator (1 = gap).  This script writes a parallel product
where gap cells are replaced by the feature's native ocean mean so that, after
z-scoring with the same mean/std, the value channel is exactly 0 at gaps.
Mask indicator channels are unchanged (still 1 at gaps).

Model-facing convention after ``normalize_and_apply_feature_mask``:
  - value ≈ 0 and mask = 1  →  gap (no imputed numeric information)
  - value = z-score and mask = 0  →  native observation

Outputs (does not overwrite ``*_men_masked``):
  data/interim/inputfeatures/SedTOCFeaturesnoNAN_{MAR,TOC}_{CS,DO}_men_masked_zero/
  data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked_zero/

Usage:
  python3 scripts/build_zeroed_masked_features.py
  python3 scripts/build_zeroed_masked_features.py --skip-world
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from mask_gap_core import load_training_mask_stack  # noqa: E402

N_ORIG = 113
CHUNK_ROWS = 6
N_COLS = 4320
N_ROWS = 2160
N_CHUNKS = N_ROWS // CHUNK_ROWS

TRAINING_PAIRS = [
    (
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked",
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked_zero",
    ),
    (
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked",
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked_zero",
    ),
    (
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked",
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked_zero",
    ),
    (
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked",
        ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked_zero",
    ),
]
WORLD_SRC = ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
WORLD_DST = ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked_zero"

COPY_SITE_FILES = (
    "numpy_lat.npy",
    "numpy_lon.npy",
    "numpy_labels.npy",
    "features_mean.npy",
    "features_std.npy",
)


def zero_gap_values(values: np.ndarray, masks: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Replace gap cells (mask > 0) with native mean → z-score becomes 0."""
    out = values.astype(np.float32, copy=True)
    if out.shape[1] != len(means):
        raise ValueError(f"value cols {out.shape[1]} != means len {len(means)}")
    if masks.shape != out.shape:
        raise ValueError(f"mask shape {masks.shape} != values shape {out.shape}")
    gap = masks > 0
    # Broadcast means across rows where gap is True.
    out[gap] = np.broadcast_to(means.astype(np.float32), out.shape)[gap]
    return out


def rebuild_training(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  skip missing {src.name}")
        return
    means = np.load(src / "features_mean.npy").astype(np.float64)
    feats = np.load(src / "numpy_features.npy")
    n = len(means)
    if feats.shape[1] != 2 * n:
        raise ValueError(f"{src.name}: expected {2 * n} cols, got {feats.shape[1]}")
    values = feats[:, :n]
    masks = feats[:, n:]
    n_gap = int((masks > 0).sum())
    zeroed = zero_gap_values(values, masks, means)
    out = np.concatenate([zeroed, masks.astype(np.float32)], axis=1)

    dst.mkdir(parents=True, exist_ok=True)
    for name in COPY_SITE_FILES:
        shutil.copy2(src / name, dst / name)
    np.save(dst / "numpy_features.npy", out)

    # Spot-check: z-score at gaps should be ~0.
    z = (zeroed.astype(np.float64) - means) / np.load(src / "features_std.npy").astype(np.float64)
    gap_z = z[masks > 0]
    max_abs = float(np.max(np.abs(gap_z))) if gap_z.size else 0.0
    print(
        f"  {dst.name}: {out.shape}, gap entries={n_gap:,}, "
        f"max |z| at gaps={max_abs:.3e}"
    )


def rebuild_world(means: np.ndarray) -> None:
    src_files = sorted(p for p in WORLD_SRC.iterdir() if p.suffix == ".npy")
    if len(src_files) != N_CHUNKS:
        raise ValueError(f"Expected {N_CHUNKS} world chunks, got {len(src_files)}")
    WORLD_DST.mkdir(parents=True, exist_ok=True)
    n = len(means)
    for i, src in enumerate(src_files):
        block = np.load(src)
        if block.shape[1] != 2 * n:
            raise ValueError(f"{src.name}: expected {2 * n} cols, got {block.shape[1]}")
        values = block[:, :n]
        masks = block[:, n:]
        zeroed = zero_gap_values(values, masks, means)
        np.save(WORLD_DST / src.name, np.concatenate([zeroed, masks.astype(np.float32)], axis=1))
        if i % 60 == 0:
            print(f"  world chunk {i + 1}/{N_CHUNKS}")
    print(f"  wrote {len(src_files)} chunks -> {WORLD_DST.name}")


def verify() -> None:
    """Confirm gap z-scores are ~0 and masks match the source product."""
    src = TRAINING_PAIRS[0][0]
    dst = TRAINING_PAIRS[0][1]
    means = np.load(dst / "features_mean.npy").astype(np.float64)
    stds = np.load(dst / "features_std.npy").astype(np.float64)
    a = np.load(src / "numpy_features.npy")
    b = np.load(dst / "numpy_features.npy")
    n = len(means)
    assert np.array_equal(a[:, n:], b[:, n:]), "masks should be identical"
    gap = a[:, n:] > 0
    z = (b[:, :n].astype(np.float64) - means) / stds
    assert float(np.max(np.abs(z[gap]))) < 1e-5, "gap z-scores should be ~0"
    # Native cells unchanged.
    native = ~gap
    assert np.allclose(a[:, :n][native], b[:, :n][native], equal_nan=True)
    print("  verify CS training: masks identical, gap z≈0, native values unchanged")

    mask_stack = load_training_mask_stack()
    chunk0_src = np.load(sorted(WORLD_SRC.glob("*.npy"))[0])
    chunk0_dst = np.load(sorted(WORLD_DST.glob("*.npy"))[0])
    assert np.array_equal(chunk0_src[:, n:], chunk0_dst[:, n:])
    # First 6 rows of mask stack should match chunk 0 mask channels.
    world_msk = chunk0_dst[:, n:].reshape(CHUNK_ROWS, N_COLS, n)
    assert np.array_equal(mask_stack[:CHUNK_ROWS, :, :], world_msk)
    print("  verify world chunk 0: masks match source + interpolation_mask_stack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-world", action="store_true", help="Training arrays only")
    args = parser.parse_args()

    print("Rebuilding training site arrays (imputed gaps -> native mean)...")
    means = None
    for src, dst in TRAINING_PAIRS:
        rebuild_training(src, dst)
        if means is None and dst.exists():
            means = np.load(dst / "features_mean.npy").astype(np.float64)

    if means is None:
        raise FileNotFoundError("No masked training dirs found to convert")

    if not args.skip_world:
        print("Rebuilding world chunks...")
        rebuild_world(means)

    print("Verification...")
    if not args.skip_world:
        verify()
    else:
        src = TRAINING_PAIRS[0][0]
        dst = TRAINING_PAIRS[0][1]
        if dst.exists():
            a = np.load(src / "numpy_features.npy")
            b = np.load(dst / "numpy_features.npy")
            n = len(means)
            gap = a[:, n:] > 0
            z = (b[:, :n].astype(np.float64) - means) / np.load(dst / "features_std.npy").astype(
                np.float64
            )
            print(
                f"  CS: max |z| at gaps={float(np.max(np.abs(z[gap]))):.3e}, "
                f"masks equal={np.array_equal(a[:, n:], b[:, n:])}"
            )
    print("Done.")


if __name__ == "__main__":
    main()
