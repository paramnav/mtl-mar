"""
NN-imputed (values-only) feature loading for mtl-mar notebooks.

Paper pipeline uses ``*_men_masked`` products (113 value channels + 113 mask
bits) but models see **values only** (113-d NN-imputed predictors). Prefer
local ``mtl-mar/data`` paths; fall back to a sibling ``nn4sedratetoc`` tree
when developing before the Zenodo package includes the imputed products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from utils import DATA_INTERIM, PROJECT_ROOT

N_ORIG_FEATURES = 113
TEST_SIZE_RATIO = 0.25
RANDOM_SEED = 42
N_BINS = 10

# Optional development fallback (same machine as nn4sedratetoc)
_NN4_ROOT = PROJECT_ROOT.parent / "nn4sedratetoc"


def _first_existing(candidates: Iterable[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    searched = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find NN-imputed feature directory. Searched:\n" + searched
    )


def resolve_input_dir(task: str, region: str) -> Path:
    """Return CS/DO inputfeature dir for MAR or TOC (``*_men_masked``)."""
    name = f"SedTOCFeaturesnoNAN_{task}_{region}_men_masked"
    return _first_existing(
        [
            DATA_INTERIM / "inputfeatures" / name,
            _NN4_ROOT / "data" / "interim" / "inputfeatures" / name,
        ]
    )


def resolve_world_dir() -> Path:
    name = "WorldFeaturesAllnoNaN_SedTOC_predict_men_masked"
    return _first_existing(
        [
            DATA_INTERIM / "worldfeatures" / name,
            _NN4_ROOT / "data" / "interim" / "worldfeatures" / name,
        ]
    )


def values_only(features: np.ndarray, n_orig: int = N_ORIG_FEATURES) -> np.ndarray:
    """Keep the 113 NN-imputed value channels (drop mask bits if present)."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features, got shape {features.shape}")
    if features.shape[1] == n_orig:
        return features.astype(np.float64)
    if features.shape[1] == 2 * n_orig:
        return features[:, :n_orig].astype(np.float64)
    if features.shape[1] > n_orig:
        # Legacy / partial products: take leading value block
        return features[:, :n_orig].astype(np.float64)
    raise ValueError(
        f"Unexpected feature width {features.shape[1]}; expected {n_orig} or {2 * n_orig}"
    )


def normalize_world_chunk(
    features: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_orig: int = N_ORIG_FEATURES,
) -> np.ndarray:
    """Z-score value channels of a world feature chunk (values-only)."""
    values = values_only(features, n_orig=n_orig)
    values = np.divide(values - x_mean, x_std)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def values_and_masks(features: np.ndarray, n_orig: int = N_ORIG_FEATURES) -> np.ndarray:
    """Keep 113 value channels + 113 mask bits (226-d) for attention models."""
    features = np.asarray(features)
    if features.ndim != 2:
        raise ValueError(f"Expected 2-D features, got shape {features.shape}")
    if features.shape[1] == 2 * n_orig:
        return features.astype(np.float64)
    if features.shape[1] == n_orig:
        # No masks on disk — synthesise all-observed indicators.
        masks = np.zeros((features.shape[0], n_orig), dtype=np.float64)
        return np.concatenate([features.astype(np.float64), masks], axis=1)
    raise ValueError(
        f"Unexpected feature width {features.shape[1]}; expected {n_orig} or {2 * n_orig}"
    )


def normalize_world_chunk_attention(
    features: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_orig: int = N_ORIG_FEATURES,
) -> np.ndarray:
    """Z-score value channels; leave mask bits unchanged (attention / 226-d)."""
    feats = values_and_masks(features, n_orig=n_orig)
    values = feats[:, :n_orig]
    masks = feats[:, n_orig:]
    mean = x_mean[:n_orig]
    std = x_std[:n_orig]
    values = np.divide(values - mean, std)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    masks = np.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate([values, masks], axis=1)


def preprocess_attention_features(
    features: np.ndarray,
    labels: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    n_orig: int = N_ORIG_FEATURES,
):
    """Drop NaN rows, z-score values, keep mask channels (no column dropping)."""
    feats = values_and_masks(features, n_orig=n_orig)
    values = feats[:, :n_orig]
    masks = feats[:, n_orig:]
    valid = ~(np.isnan(values).any(axis=1) | np.isnan(labels))
    values = values[valid]
    masks = masks[valid]
    labels = labels[valid]
    lats = lats[valid]
    lons = lons[valid]
    mean = x_mean[:n_orig]
    std = x_std[:n_orig]
    values = np.divide(values - mean, std)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    features_out = np.concatenate([values, masks], axis=1)
    return features_out, labels, lats, lons


def load_region_arrays(
    task: str,
    region: str,
    preprocess_fn,
    n_orig: int = N_ORIG_FEATURES,
):
    """Load one region, keep values-only, preprocess with *preprocess_fn*."""
    feature_path = resolve_input_dir(task, region)
    features = np.load(feature_path / "numpy_features.npy")
    labels = np.load(feature_path / "numpy_labels.npy")
    x_mean = np.load(feature_path / "features_mean.npy").astype(float)
    x_std = np.load(feature_path / "features_std.npy").astype(float)
    lats = np.load(feature_path / "numpy_lat.npy")
    lons = np.load(feature_path / "numpy_lon.npy")

    features = values_only(features, n_orig=n_orig)
    if x_mean.shape[0] != features.shape[1]:
        x_mean = x_mean[: features.shape[1]]
        x_std = x_std[: features.shape[1]]

    features, labels, lats, lons = preprocess_fn(
        features, labels, lats, lons, x_mean, x_std
    )
    return {
        "features": features,
        "labels": labels,
        "lats": lats,
        "lons": lons,
        "x_mean": x_mean,
        "x_std": x_std,
        "path": feature_path,
    }


def stratified_split(features, labels, test_size_ratio=TEST_SIZE_RATIO, random_seed=RANDOM_SEED):
    """75/25 stratified split by MAR/TOC deciles (paper recipe)."""
    from sklearn.model_selection import train_test_split

    n_bins = min(N_BINS, max(2, len(np.unique(labels))))
    y_binned = pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")
    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=test_size_ratio,
        random_state=random_seed,
        stratify=y_binned,
    )
    return train_features, train_labels, test_features, test_labels


def load_mar_cs_do(preprocess_fn, test_size_ratio=TEST_SIZE_RATIO, random_seed=RANDOM_SEED):
    """Load CS+DO MAR with region-wise stratified 75/25 splits, then concatenate."""
    world_dir = resolve_world_dir()
    world_files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
    world_dim = values_only(np.load(world_files[0])).shape[1]

    regions = {}
    for region in ("DO", "CS"):
        regions[region] = load_region_arrays("MAR", region, preprocess_fn)

    if regions["DO"]["features"].shape[1] != world_dim:
        raise ValueError(
            f"Feature dim mismatch: world={world_dim}, "
            f"DO={regions['DO']['features'].shape[1]}"
        )
    if regions["CS"]["features"].shape[1] != world_dim:
        raise ValueError(
            f"Feature dim mismatch: world={world_dim}, "
            f"CS={regions['CS']['features'].shape[1]}"
        )

    splits = {}
    for region, data in regions.items():
        tr_x, tr_y, te_x, te_y = stratified_split(
            data["features"], data["labels"], test_size_ratio, random_seed
        )
        splits[region] = {
            "train_features": tr_x,
            "train_labels": tr_y,
            "test_features": te_x,
            "test_labels": te_y,
        }

    # Use DO normalization for global maps (same convention as nn4sedratetoc)
    return {
        "world_dir": world_dir,
        "world_dim": world_dim,
        "x_mean": regions["DO"]["x_mean"],
        "x_std": regions["DO"]["x_std"],
        "paths": {"DO": regions["DO"]["path"], "CS": regions["CS"]["path"]},
        "train_features": np.concatenate(
            [splits["CS"]["train_features"], splits["DO"]["train_features"]], axis=0
        ),
        "train_labels": np.concatenate(
            [splits["CS"]["train_labels"], splits["DO"]["train_labels"]], axis=0
        ),
        "test_features": np.concatenate(
            [splits["CS"]["test_features"], splits["DO"]["test_features"]], axis=0
        ),
        "test_labels": np.concatenate(
            [splits["CS"]["test_labels"], splits["DO"]["test_labels"]], axis=0
        ),
        "features": np.concatenate(
            [regions["CS"]["features"], regions["DO"]["features"]], axis=0
        ),
        "labels": np.concatenate(
            [regions["CS"]["labels"], regions["DO"]["labels"]], axis=0
        ),
        "n_features": world_dim,
    }


def load_toc_cs_do(preprocess_fn, test_size_ratio=TEST_SIZE_RATIO, random_seed=RANDOM_SEED):
    """Load CS+DO TOC with the same stratified split recipe as MAR."""
    regions = {
        region: load_region_arrays("TOC", region, preprocess_fn)
        for region in ("DO", "CS")
    }
    splits = {}
    for region, data in regions.items():
        tr_x, tr_y, te_x, te_y = stratified_split(
            data["features"], data["labels"], test_size_ratio, random_seed
        )
        splits[region] = {
            "train_features": tr_x,
            "train_labels": tr_y,
            "test_features": te_x,
            "test_labels": te_y,
        }

    return {
        "train_features": np.concatenate(
            [splits["CS"]["train_features"], splits["DO"]["train_features"]], axis=0
        ),
        "train_labels": np.concatenate(
            [splits["CS"]["train_labels"], splits["DO"]["train_labels"]], axis=0
        ),
        "test_features": np.concatenate(
            [splits["CS"]["test_features"], splits["DO"]["test_features"]], axis=0
        ),
        "test_labels": np.concatenate(
            [splits["CS"]["test_labels"], splits["DO"]["test_labels"]], axis=0
        ),
        "features": np.concatenate(
            [regions["CS"]["features"], regions["DO"]["features"]], axis=0
        ),
        "labels": np.concatenate(
            [regions["CS"]["labels"], regions["DO"]["labels"]], axis=0
        ),
        "paths": {"DO": regions["DO"]["path"], "CS": regions["CS"]["path"]},
    }


def _load_region_arrays_attention(task: str, region: str, n_orig: int = N_ORIG_FEATURES):
    """Load one region keeping value+mask channels (226-d)."""
    feature_path = resolve_input_dir(task, region)
    features = np.load(feature_path / "numpy_features.npy")
    labels = np.load(feature_path / "numpy_labels.npy")
    x_mean = np.load(feature_path / "features_mean.npy").astype(float)
    x_std = np.load(feature_path / "features_std.npy").astype(float)
    lats = np.load(feature_path / "numpy_lat.npy")
    lons = np.load(feature_path / "numpy_lon.npy")
    features, labels, lats, lons = preprocess_attention_features(
        features, labels, lats, lons, x_mean, x_std, n_orig=n_orig
    )
    return {
        "features": features,
        "labels": labels,
        "lats": lats,
        "lons": lons,
        "x_mean": x_mean[:n_orig],
        "x_std": x_std[:n_orig],
        "path": feature_path,
    }


def load_mar_cs_do_attention(
    test_size_ratio=TEST_SIZE_RATIO,
    random_seed=RANDOM_SEED,
    n_orig: int = N_ORIG_FEATURES,
):
    """CS+DO MAR with value+mask channels for attention models (226-d)."""
    world_dir = resolve_world_dir()
    world_files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
    world_dim = values_and_masks(np.load(world_files[0]), n_orig=n_orig).shape[1]

    regions = {
        region: _load_region_arrays_attention("MAR", region, n_orig=n_orig)
        for region in ("DO", "CS")
    }
    for region, data in regions.items():
        if data["features"].shape[1] != world_dim:
            raise ValueError(
                f"Feature dim mismatch: world={world_dim}, "
                f"{region}={data['features'].shape[1]}"
            )

    splits = {}
    for region, data in regions.items():
        tr_x, tr_y, te_x, te_y = stratified_split(
            data["features"], data["labels"], test_size_ratio, random_seed
        )
        splits[region] = {
            "train_features": tr_x,
            "train_labels": tr_y,
            "test_features": te_x,
            "test_labels": te_y,
        }

    return {
        "world_dir": world_dir,
        "world_dim": world_dim,
        "x_mean": regions["DO"]["x_mean"],
        "x_std": regions["DO"]["x_std"],
        "paths": {"DO": regions["DO"]["path"], "CS": regions["CS"]["path"]},
        "train_features": np.concatenate(
            [splits["CS"]["train_features"], splits["DO"]["train_features"]], axis=0
        ),
        "train_labels": np.concatenate(
            [splits["CS"]["train_labels"], splits["DO"]["train_labels"]], axis=0
        ),
        "test_features": np.concatenate(
            [splits["CS"]["test_features"], splits["DO"]["test_features"]], axis=0
        ),
        "test_labels": np.concatenate(
            [splits["CS"]["test_labels"], splits["DO"]["test_labels"]], axis=0
        ),
        "features": np.concatenate(
            [regions["CS"]["features"], regions["DO"]["features"]], axis=0
        ),
        "labels": np.concatenate(
            [regions["CS"]["labels"], regions["DO"]["labels"]], axis=0
        ),
        "n_features": world_dim,
        "n_value_features": n_orig,
    }


def load_toc_cs_do_attention(
    test_size_ratio=TEST_SIZE_RATIO,
    random_seed=RANDOM_SEED,
    n_orig: int = N_ORIG_FEATURES,
):
    """CS+DO TOC with value+mask channels for attention MTL (226-d)."""
    regions = {
        region: _load_region_arrays_attention("TOC", region, n_orig=n_orig)
        for region in ("DO", "CS")
    }
    splits = {}
    for region, data in regions.items():
        tr_x, tr_y, te_x, te_y = stratified_split(
            data["features"], data["labels"], test_size_ratio, random_seed
        )
        splits[region] = {
            "train_features": tr_x,
            "train_labels": tr_y,
            "test_features": te_x,
            "test_labels": te_y,
        }

    return {
        "train_features": np.concatenate(
            [splits["CS"]["train_features"], splits["DO"]["train_features"]], axis=0
        ),
        "train_labels": np.concatenate(
            [splits["CS"]["train_labels"], splits["DO"]["train_labels"]], axis=0
        ),
        "test_features": np.concatenate(
            [splits["CS"]["test_features"], splits["DO"]["test_features"]], axis=0
        ),
        "test_labels": np.concatenate(
            [splits["CS"]["test_labels"], splits["DO"]["test_labels"]], axis=0
        ),
        "features": np.concatenate(
            [regions["CS"]["features"], regions["DO"]["features"]], axis=0
        ),
        "labels": np.concatenate(
            [regions["CS"]["labels"], regions["DO"]["labels"]], axis=0
        ),
        "paths": {"DO": regions["DO"]["path"], "CS": regions["CS"]["path"]},
        "n_features": regions["CS"]["features"].shape[1],
        "n_value_features": n_orig,
    }


def predict_global_map_sklearn(
    model,
    world_dir: Path,
    x_mean: np.ndarray,
    x_std: np.ndarray,
    land_map_path: Path,
    out_path: Path,
    chunk_shape: Sequence[int] = (6, 4320),
) -> np.ndarray:
    """Chunked world prediction for sklearn-like models; land masked to 0."""
    files = sorted(p for p in Path(world_dir).iterdir() if p.suffix == ".npy")
    prediction_map = None
    for i, file in enumerate(files):
        feats = normalize_world_chunk(np.load(file), x_mean, x_std)
        preds = model.predict(feats).reshape(chunk_shape)
        prediction_map = preds if prediction_map is None else np.append(prediction_map, preds, axis=0)
        if (i + 1) % 60 == 0:
            print(f"  predicted {i + 1}/{len(files)} chunks")

    land_map = np.load(land_map_path)
    land_map = np.where(np.isnan(land_map), 1, land_map).astype(bool)
    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))
    prediction_map[land_map] = 0

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, prediction_map)
    print(f"Saved prediction map: {out_path}")
    return prediction_map
