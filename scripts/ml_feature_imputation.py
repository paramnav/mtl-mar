"""Shared ocean-gap cube / native-mask helpers for NN feature imputation.

Paper gap filling uses ``nn_feature_imputation.py`` (per-feature MLP). This
module holds the common data contract used by that imputer:

  * Load the native feature stack as an (H, W, F) cube
  * Build per-feature native-ocean masks
  * Mask predictors to NaN wherever a predictor is itself a gap
  * Clip rules and circular (wave-direction) helpers

HistGradientBoosting imputation was removed; use ``harmonize_ocean_gaps.py``
(NN only).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mask_gap_core import (
    is_garlan_feature,
    is_rivermouth_feature,
    load_grid_with_coords,
    resolve_r50_base,
)

try:
    from feature_update_core import FEATURE_STACK
except ImportError:
    from pathlib import Path

    FEATURE_STACK = Path(__file__).resolve().parents[1] / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"


DEFAULT_MAX_TRAIN_SAMPLES = 400_000
DEFAULT_MIN_NATIVE_PREDICTORS = 5
GEO_PREDICTOR_NAMES = ("_lat_deg", "_lon_deg")


@dataclass(frozen=True)
class ClipRule:
    lo: float | None = None
    hi: float | None = None


def _clip_rule_for(fname: str) -> ClipRule | None:
    if fname.startswith("probabilities_"):
        return ClipRule(0.0, 1.0)
    if "WAVE_HEIGHT" in fname or "WAVE_PERIOD" in fname:
        return ClipRule(0.0, None)
    if "CRUST_AGE" in fname:
        return ClipRule(0.0, 340.0)
    if fname.startswith("TOU_"):
        return ClipRule(0.0, None)
    if "litho_maps_type" in fname:
        return ClipRule(0.0, 1.0)
    return None


def _is_circular_direction(fname: str) -> bool:
    return "WAVE_DIRECTION" in fname


def _predictor_indices(target_idx: int, names: list[str]) -> list[int]:
    """Indices of predictors: all features except target and its r50km companions."""
    name_set = set(names)
    target_name = names[target_idx]
    blocked = {target_name}
    for j, fname in enumerate(names):
        if j == target_idx:
            continue
        base = resolve_r50_base(fname, name_set)
        if base == target_name:
            blocked.add(fname)
    return [j for j, fname in enumerate(names) if fname not in blocked]


def _build_geo_grids(lats: np.ndarray, lons: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    lat_2d = np.broadcast_to(lats[:, None], shape)
    lon_2d = np.broadcast_to(lons[None, :], shape)
    return lat_2d.astype(np.float32), lon_2d.astype(np.float32)


def load_stack_cube(
    names: list[str],
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Return (H, W, n_features) float32 array."""
    h, w = ocean.shape
    cube = np.full((h, w, len(names)), np.nan, dtype=np.float32)
    for j, fname in enumerate(names):
        arr, _, _ = load_grid_with_coords(FEATURE_STACK / fname)
        if arr.shape != (h, w):
            raise ValueError(f"Shape mismatch for {fname}: {arr.shape} vs {(h, w)}")
        cube[:, :, j] = np.asarray(arr, dtype=np.float32)
    return cube


def native_mask_for_feature(
    grid: np.ndarray,
    gap: np.ndarray,
    ocean: np.ndarray,
) -> np.ndarray:
    """True where the feature is a native ocean observation."""
    return ocean & (~gap) & np.isfinite(grid)


def mask_predictors_to_native(
    cube: np.ndarray,
    gap_stack: np.ndarray,
    ocean: np.ndarray,
    pred_indices: list[int],
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flattened (n_ocean, n_pred+2) matrix with NaN for non-native predictors."""
    ocean_idx = np.flatnonzero(ocean.ravel())
    n_pred = len(pred_indices)
    n_cols = n_pred + len(GEO_PREDICTOR_NAMES)
    out = np.empty((ocean_idx.size, n_cols), dtype=np.float32)
    flat_cube = cube.reshape(-1, cube.shape[-1])
    flat_gaps = gap_stack.reshape(-1, gap_stack.shape[-1])
    for k, j in enumerate(pred_indices):
        vals = flat_cube[:, j].astype(np.float64)
        missing = flat_gaps[:, j] | ~np.isfinite(vals)
        vals[missing] = np.nan
        out[:, k] = vals[ocean_idx]
    out[:, n_pred] = lat_grid.ravel()[ocean_idx]
    out[:, n_pred + 1] = lon_grid.ravel()[ocean_idx]
    return out, ocean_idx


def count_native_predictors(x: np.ndarray) -> np.ndarray:
    return np.sum(np.isfinite(x), axis=1)


def _clip_values(values: np.ndarray, rule: ClipRule | None) -> np.ndarray:
    if rule is None:
        return values
    out = values.copy()
    if rule.lo is not None:
        out = np.maximum(out, rule.lo)
    if rule.hi is not None:
        out = np.minimum(out, rule.hi)
    return out


def base_features_with_gaps(
    names: list[str],
    gap_by_name: dict[str, np.ndarray],
    ocean: np.ndarray,
) -> list[str]:
    """Base (non-r50km) features that need gap filling, excluding rivermouth."""
    name_set = set(names)
    out: list[str] = []
    for fname in names:
        if resolve_r50_base(fname, name_set) is not None:
            continue
        if is_rivermouth_feature(fname) or is_garlan_feature(fname):
            continue
        gap = gap_by_name.get(fname)
        if gap is None or not (gap & ocean).any():
            continue
        out.append(fname)
    return out
