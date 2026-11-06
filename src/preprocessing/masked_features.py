"""Masked-feature loading for sedimentation / TOC model runs.

On-disk products (226 columns = 113 values + 113 per-feature mask bits):

* ``*_men_masked`` — gap cells keep NN/ML-imputed values (default).
* ``*_men_masked_zero`` — gap cells hold the native ocean mean so z-scores are
  exactly 0 at gaps (no imputed numeric information). Built by
  ``scripts/build_zeroed_masked_features.py``.

Select the product / representation with env ``MASKED_FEATURE_VARIANT``:

  - ``imputed`` (default) → ``*_men_masked``, all 113 mask bits kept (226-dim)
  - ``zero`` → ``*_men_masked_zero``, all 113 mask bits kept (226-dim)
  - ``summary`` → ``*_men_masked`` values, but masks collapsed to one scalar
    ``imputation_fraction = mean(mask bits)`` (114-dim). Outputs tagged
    ``masked_summary`` so they do not overwrite full-mask runs.
  - ``values_only`` / ``nomask`` → ``*_men_masked`` NN-imputed values only
    (113-dim; mask channels dropped). Tag ``nomask``.
  - ``diesing_zero`` → like ``values_only``, but Diesing lithology probability
    channels are set to 0 where elevation > -500 m (shelf / shallow). Tag
    ``diesing_zero``.
  - ``attention`` → same data as ``imputed`` (226-dim values+masks); output
    tag ``attention`` so attention-model maps do not overwrite MLP runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

N_ORIG_FEATURES = 113
SHALLOW_ELEV_M = -500.0
ELEVATION_FEATURE = "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd"
DIESING_FEATURES = (
    "probabilities_Calc.Sed_mod.nc",
    "probabilities_Clay_mod.nc",
    "probabilities_Dia.Ooze_mod.nc",
    "probabilities_Lith.Sed_mod.nc",
)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # mtl-mar/
MAR_MASKED_DIRS = {
    "CS": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked",
    "DO": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked",
}
TOC_MASKED_DIRS = {
    "CS": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked",
    "DO": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked",
}
MAR_MASKED_ZERO_DIRS = {
    "CS": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_CS_men_masked_zero",
    "DO": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_DO_men_masked_zero",
}
TOC_MASKED_ZERO_DIRS = {
    "CS": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_CS_men_masked_zero",
    "DO": PROJECT_ROOT / "data/interim/inputfeatures/SedTOCFeaturesnoNAN_TOC_DO_men_masked_zero",
}
WORLD_MASKED_DIR = (
    PROJECT_ROOT / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
)
WORLD_MASKED_ZERO_DIR = (
    PROJECT_ROOT
    / "data/interim/worldfeatures/WorldFeaturesAllnoNaN_SedTOC_predict_men_masked_zero"
)


def training_feature_names() -> list[str]:
    """113 training predictor names (leading elevation-derivative reference dropped)."""
    names = [
        ln.strip()
        for ln in (
            PROJECT_ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt"
        )
        .read_text()
        .splitlines()
        if ln.strip()
    ]
    if names and names[0].startswith("DERIVATIVE_GL_ELEVATION"):
        return names[1:]
    return names


def diesing_column_indices(n_orig: int = N_ORIG_FEATURES) -> tuple[int, list[int]]:
    """Return (elevation_col, diesing_cols) into the 113-d value block."""
    names = training_feature_names()
    if len(names) != n_orig:
        raise ValueError(f"expected {n_orig} training names, got {len(names)}")
    try:
        elev_idx = names.index(ELEVATION_FEATURE)
    except ValueError as exc:
        raise ValueError(f"elevation feature {ELEVATION_FEATURE!r} missing") from exc
    diesing_idxs = []
    for fname in DIESING_FEATURES:
        try:
            diesing_idxs.append(names.index(fname))
        except ValueError as exc:
            raise ValueError(f"Diesing feature {fname!r} missing") from exc
    return elev_idx, diesing_idxs


def masked_feature_variant() -> str:
    """Return active variant from ``MASKED_FEATURE_VARIANT``."""
    raw = os.environ.get("MASKED_FEATURE_VARIANT", "imputed").strip().lower()
    if raw in ("", "imputed", "mask", "masked"):
        return "imputed"
    if raw in ("zero", "zerofill", "masked_zero", "meanfill"):
        return "zero"
    if raw in ("summary", "summary_mask", "masked_summary", "frac", "fraction"):
        return "summary"
    if raw in ("values_only", "nomask", "no_mask", "unmasked", "nn_only"):
        return "values_only"
    if raw in (
        "diesing_zero",
        "diesing_shallow_zero",
        "nomask_diesing_zero",
        "values_diesing_zero",
    ):
        return "diesing_zero"
    if raw in ("attention", "attn", "masked_attention"):
        return "attention"
    if raw in ("cross_attention", "crossattn", "cross_attn", "xattn"):
        return "cross_attention"
    if raw in ("ft_transformer", "fttransformer", "fttrans", "ft-transformer"):
        return "ft_transformer"
    raise ValueError(
        f"Unknown MASKED_FEATURE_VARIANT={raw!r}; use 'imputed', 'zero', "
        f"'summary', 'values_only', 'diesing_zero', 'attention', "
        f"'cross_attention', or 'ft_transformer'"
    )


def drops_mask_channels() -> bool:
    """True when the model sees value channels only (no mask bits)."""
    return masked_feature_variant() in ("values_only", "diesing_zero")


def masked_dir_suffix() -> str:
    """Directory suffix for on-disk products (summary/values reuse imputed dirs)."""
    return "men_masked_zero" if masked_feature_variant() == "zero" else "men_masked"


def masked_output_tag() -> str:
    """Filename tag for prediction maps / figures."""
    v = masked_feature_variant()
    if v == "zero":
        return "masked_zero"
    if v == "summary":
        return "masked_summary"
    if v == "values_only":
        return "nomask"
    if v == "diesing_zero":
        return "diesing_zero"
    if v == "attention":
        return "attention"
    if v == "cross_attention":
        return "cross_attention"
    if v == "ft_transformer":
        return "ft_transformer"
    return "masked"


def active_mar_dirs() -> dict[str, Path]:
    return MAR_MASKED_ZERO_DIRS if masked_feature_variant() == "zero" else MAR_MASKED_DIRS


def active_toc_dirs() -> dict[str, Path]:
    return TOC_MASKED_ZERO_DIRS if masked_feature_variant() == "zero" else TOC_MASKED_DIRS


def active_world_dir() -> Path:
    return WORLD_MASKED_ZERO_DIR if masked_feature_variant() == "zero" else WORLD_MASKED_DIR


def active_mar_dir(region: str) -> Path:
    return active_mar_dirs()[region]


def active_toc_dir(region: str) -> Path:
    return active_toc_dirs()[region]


def variant_data_path(path) -> Path:
    """Rewrite a hardcoded ``*_men_masked*`` path for the active variant."""
    text = str(path)
    if masked_feature_variant() != "zero":
        # imputed and summary both read *_men_masked
        return Path(text)
    if "men_masked_zero" in text:
        return Path(text)
    return Path(text.replace("men_masked", "men_masked_zero"))


def _retag_output_path(text: str, tag: str) -> Path:
    """Replace ``_masked`` filename tags with ``_{tag}`` (e.g. masked_summary)."""
    if f"_{tag}." in text or text.endswith(f"_{tag}") or f"_{tag}_" in text:
        return Path(text)
    if text.endswith("_masked.npy"):
        return Path(text[:-11] + f"_{tag}.npy")
    if text.endswith("_masked.png"):
        return Path(text[:-11] + f"_{tag}.png")
    if text.endswith("_masked"):
        return Path(text[:-7] + f"_{tag}")
    if "_masked_" in text:
        return Path(text.replace("_masked_", f"_{tag}_"))
    return Path(text)


def variant_output_path(path) -> Path:
    """Rewrite prediction-map paths so variants do not overwrite each other."""
    text = str(path)
    v = masked_feature_variant()
    if v == "imputed":
        return Path(text)
    if v == "zero":
        return _retag_output_path(text, "masked_zero")
    if v == "summary":
        return _retag_output_path(text, "masked_summary")
    if v == "values_only":
        return _retag_output_path(text, "nomask")
    if v == "diesing_zero":
        return _retag_output_path(text, "diesing_zero")
    if v == "attention":
        return _retag_output_path(text, "attention")
    if v == "cross_attention":
        return _retag_output_path(text, "cross_attention")
    if v == "ft_transformer":
        return _retag_output_path(text, "ft_transformer")
    return Path(text)


def collapse_mask_indicators(indicators: np.ndarray) -> np.ndarray:
    """Collapse per-feature masks to one imputation-fraction column (n, 1)."""
    if indicators.ndim != 2:
        raise ValueError(f"indicators must be 2-D, got shape {indicators.shape}")
    if indicators.shape[1] == 1:
        return indicators.astype(np.float64)
    frac = indicators.astype(np.float64).mean(axis=1, keepdims=True)
    return frac


def maybe_collapse_masks(indicators: np.ndarray) -> np.ndarray:
    """Apply summary collapse when ``MASKED_FEATURE_VARIANT=summary``."""
    if masked_feature_variant() == "summary":
        return collapse_mask_indicators(indicators)
    return indicators


def load_filtered_mask_stack():
    """Load 113-column mask stack aligned with training predictors.

    Uses ``interpolation_mask_stack.npy`` — the same indicators written into
    world cubes and shown in ``plot_masked_feature_maps``.
    """
    import sys

    scripts = PROJECT_ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from mask_gap_core import load_training_mask_stack

    return load_training_mask_stack()


def indicators_at_latlon(lats, lons, mask_stack, valid_cols=None):
    lat_grid = np.linspace(-90, 90, 2160)
    lon_grid = np.linspace(-180, 180, 4320)
    out = np.zeros((len(lats), mask_stack.shape[2]), dtype=np.float64)
    for i in range(len(lats)):
        r = int(np.argmin(np.abs(lat_grid - lats[i])))
        c = int(np.argmin(np.abs(lon_grid - lons[i])))
        out[i] = mask_stack[r, c, :]
    if valid_cols is not None:
        out = out[:, ~valid_cols]
    return out


def _zero_gap_values(values, indicators):
    """Force z-scored gap cells to 0 when running the zero-gap ablation."""
    if masked_feature_variant() != "zero":
        return values
    # indicators may already be collapsed in summary mode, but zero never collapses
    if indicators.shape[1] == 1 and values.shape[1] > 1:
        raise ValueError("zero-gap variant requires per-feature mask indicators")
    return values * (1.0 - indicators)


def _zero_diesing_shallow(values: np.ndarray, n_orig: int = N_ORIG_FEATURES) -> np.ndarray:
    """Set Diesing lithology probabilities to 0 where elev > -500 m (raw units)."""
    if masked_feature_variant() != "diesing_zero":
        return values
    elev_idx, diesing_idxs = diesing_column_indices(n_orig)
    out = values.astype(np.float64, copy=True)
    shallow = out[:, elev_idx] > SHALLOW_ELEV_M
    if not np.any(shallow):
        return out
    out[np.ix_(shallow, diesing_idxs)] = 0.0
    return out


def _finalize_feature_matrix(values: np.ndarray, indicators: np.ndarray) -> np.ndarray:
    """Optionally drop / collapse mask channels per active variant."""
    if drops_mask_channels():
        return values
    indicators = maybe_collapse_masks(indicators)
    return np.concatenate([values, indicators], axis=1)


def normalize_and_apply_feature_mask(features, X_mean, X_std, n_orig=N_ORIG_FEATURES):
    """Z-score value channels; leave indicator channels unchanged.

    For ``*_men_masked`` products the value channel keeps imputed numbers.
    For ``*_men_masked_zero`` (or ``MASKED_FEATURE_VARIANT=zero``) gap cells are
    forced to z=0 so the value channel carries no imputed numeric information.
    For ``summary``, the 113 mask bits are replaced by one imputation fraction.
    For ``values_only`` / ``diesing_zero``, only the (z-scored) value channels
    are returned; ``diesing_zero`` zeros Diesing probs on the shelf first.
    """
    values = features[:, :n_orig].astype(np.float64)
    indicators = features[:, n_orig:].astype(np.float64)
    values = _zero_diesing_shallow(values, n_orig)
    values = np.divide((values - X_mean), X_std)
    values = _zero_gap_values(values, indicators)
    # World cubes keep NaNs on land / incomplete cells; trees and KNN reject them.
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return _finalize_feature_matrix(values, indicators)


def preprocess_features_labels_masked(
    features, labels, lats, lons, X_mean, X_std, mask_stack=None, n_orig=N_ORIG_FEATURES
):
    """Return z-scored values (+ mask indicators unless values-only variants).

    If *features* already has ``2 * n_orig`` columns (from
    ``rebuild_harmonized_feature_products.py``), uses the stored mask
    indicators in columns ``n_orig:``. Otherwise rebuilds indicators from
    *mask_stack* (legacy 111-column inputs).

    Duplicate sites are collapsed with the same rules as
    ``trainDNN.preprocess_features_labels`` (dedup on normalized values only).
    """
    from trainDNN import deduplicate_training_sites, preprocess_features_labels

    if features.shape[1] == 2 * n_orig:
        values = features[:, :n_orig].astype(np.float64)
        indicators = features[:, n_orig:].astype(np.float64)
        valid = ~(np.isnan(values).any(axis=1) | np.isnan(labels))
        values = values[valid]
        indicators = indicators[valid]
        labels = labels[valid]
        lats = lats[valid]
        lons = lons[valid]
        values = _zero_diesing_shallow(values, n_orig)
        values = np.divide((values - X_mean), X_std)
        values = _zero_gap_values(values, indicators)
        values, labels, lats, lons, (indicators,) = deduplicate_training_sites(
            values, labels, lats, lons, row_arrays=(indicators,)
        )
        return _finalize_feature_matrix(values, indicators), labels, lats, lons

    if mask_stack is None:
        raise ValueError("mask_stack is required when features have only value columns")

    values = features[:, :n_orig].astype(np.float64)
    values = _zero_diesing_shallow(values, n_orig)
    values_proc, labels, lats, lons = preprocess_features_labels(
        values, labels, lats, lons, X_mean, X_std
    )
    indicators = indicators_at_latlon(lats, lons, mask_stack)
    if values_proc.shape[1] != indicators.shape[1]:
        keep = min(values_proc.shape[1], indicators.shape[1])
        values_proc = values_proc[:, :keep]
        indicators = indicators[:, :keep]
    values_proc = _zero_gap_values(values_proc, indicators)
    return _finalize_feature_matrix(values_proc, indicators), labels, lats, lons


def load_masked_stats(reference_dir=None):
    """Return (X_mean, X_std) from a masked feature directory."""
    ref = reference_dir or active_mar_dirs()["CS"]
    return (
        np.load(ref / "features_mean.npy").astype(float),
        np.load(ref / "features_std.npy").astype(float),
    )


def load_masked_site_arrays(input_dir, mask_stack=None):
    """Load raw 226-col arrays from a masked training dir and preprocess them."""
    if mask_stack is None:
        mask_stack = load_filtered_mask_stack()
    X_mean, X_std = load_masked_stats(input_dir)
    features = np.load(input_dir / "numpy_features.npy")
    labels = np.load(input_dir / "numpy_labels.npy")
    lats = np.load(input_dir / "numpy_lat.npy")
    lons = np.load(input_dir / "numpy_lon.npy")
    return preprocess_features_labels_masked(
        features, labels, lats, lons, X_mean, X_std, mask_stack
    )


def _split_task_arrays(features, labels, test_size_ratio, random_seed):
    from trainDNN import split_train_test_data

    n_bins = min(10, len(np.unique(labels)))
    y_binned = pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")
    train_features, train_labels, test_features, test_labels = split_train_test_data(
        features,
        labels,
        test_size_ratio=test_size_ratio,
        random_seed=random_seed,
        stratify=y_binned,
    )
    return {
        "features": features,
        "labels": labels,
        "train_features": train_features,
        "train_labels": train_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "n_samples": len(labels),
    }


def load_region_pair_task_masked(
    region_dirs,
    test_size_ratio=0.25,
    random_seed=42,
    mask_stack=None,
):
    """Load CS+DO masked arrays, preprocess, deduplicate, and stratified-split."""
    region_arrays = {}
    for region, input_dir in region_dirs.items():
        features, labels, _, _ = load_masked_site_arrays(input_dir, mask_stack)
        region_arrays[region] = (features, labels)

    features = np.concatenate([region_arrays["CS"][0], region_arrays["DO"][0]], axis=0)
    labels = np.concatenate([region_arrays["CS"][1], region_arrays["DO"][1]], axis=0)
    return _split_task_arrays(features, labels, test_size_ratio, random_seed)


def load_mar_task_masked(test_size_ratio=0.25, random_seed=42, mask_stack=None):
    """Load MAR CS+DO for the active ``MASKED_FEATURE_VARIANT``."""
    return load_region_pair_task_masked(
        active_mar_dirs(), test_size_ratio, random_seed, mask_stack
    )


def load_toc_task_masked(test_size_ratio=0.25, random_seed=42, mask_stack=None):
    """Load TOC CS+DO for the active ``MASKED_FEATURE_VARIANT``."""
    return load_region_pair_task_masked(
        active_toc_dirs(), test_size_ratio, random_seed, mask_stack
    )


def load_mar_task_masked_zero(test_size_ratio=0.25, random_seed=42, mask_stack=None):
    """MAR task with neutral (z=0) gap values + same mask indicators."""
    return load_region_pair_task_masked(
        MAR_MASKED_ZERO_DIRS, test_size_ratio, random_seed, mask_stack
    )


def load_toc_task_masked_zero(test_size_ratio=0.25, random_seed=42, mask_stack=None):
    """TOC task with neutral (z=0) gap values + same mask indicators."""
    return load_region_pair_task_masked(
        TOC_MASKED_ZERO_DIRS, test_size_ratio, random_seed, mask_stack
    )
