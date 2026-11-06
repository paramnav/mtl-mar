"""Extract 111-feature TOC training bundles (DO + CS) for MTL pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
from utils import DATA_INTERIM, DATA_RAW  # noqa: E402
from models.train_neural_network import preprocess_features_labels, set_seed  # noqa: E402


def grid_indices(latitudes: np.ndarray, longitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest 5-arcmin grid cell for ETOPO-style 2160 x 4320 grids."""
    lat_idx = np.clip(np.rint((latitudes + 90.0) / 180.0 * 2159.0).astype(int), 0, 2159)
    lon_idx = np.clip(np.rint((longitudes + 180.0) / 360.0 * 4319.0).astype(int), 0, 4319)
    return lat_idx, lon_idx


def extract_features(df: pd.DataFrame, restrepo_files: list[str], features_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = len(df)
    n_features = len(restrepo_files)
    features = np.empty((n_samples, n_features))
    features[:] = np.nan
    x_mean = np.empty(0)
    x_std = np.empty(0)

    lat_idx, lon_idx = grid_indices(df["Latitude"].values, df["Longitude"].values)

    corrupted = 0
    for file_idx, file in enumerate(restrepo_files):
        try:
            rootgrp = xr.open_dataset(features_path / file, format="NETCDF4")
        except Exception as exc:
            print(f"Warning: could not load {file}: {exc}")
            x_mean = np.append(x_mean, np.nan)
            x_std = np.append(x_std, np.nan)
            corrupted += 1
            continue

        dataset = rootgrp.variables["data"]
        grid = np.asarray(dataset[:])
        x_mean = np.append(x_mean, np.nanmean(grid))
        x_std = np.append(x_std, np.nanstd(grid))

        values = grid[lat_idx, lon_idx]
        if file == "GL_ELEVATION_M_ASL_ETOPO2v2.5m.grd":
            values = values.astype(float)
            values[values > 5] = np.nan
        features[:, file_idx] = values

        if (file_idx + 1) % 20 == 0 or file_idx + 1 == n_features:
            print(f"  processed {file_idx + 1}/{n_features} files")

    print(f"  corrupted files: {corrupted}")
    return features, x_mean, x_std


def build_bundle(label_file: str, save_name: str) -> None:
    print(f"\n=== {save_name} ===", flush=True)
    df = pd.read_csv(DATA_RAW / "labels" / label_file)
    df = df[["Latitude", "Longitude", "TOC [%]"]]
    df = df[df["Longitude"] > -180]
    df = df.dropna(how="any")
    print(f"labels after cleaning: {len(df)}")

    feature_list_path = DATA_INTERIM / "selectedFeatureLists" / "selectedfeatures_v_men.txt"
    restrepo_files = pd.read_csv(feature_list_path, header=None).values.flatten().tolist()
    features_path = DATA_RAW / "features" / "FeaturesPhrampusLee_TOCSedRate_updated"

    features, x_mean, x_std = extract_features(df, restrepo_files, features_path)
    save_path = DATA_INTERIM / "inputfeatures" / save_name
    save_path.mkdir(parents=True, exist_ok=True)

    np.save(save_path / "numpy_features.npy", features)
    np.save(save_path / "numpy_labels.npy", df["TOC [%]"].values)
    np.save(save_path / "features_mean.npy", x_mean)
    np.save(save_path / "features_std.npy", x_std)
    np.save(save_path / "numpy_lat.npy", df["Latitude"].values)
    np.save(save_path / "numpy_lon.npy", df["Longitude"].values)
    (save_path / "feature_files.txt").write_text("\n".join(restrepo_files), encoding="utf-8")
    print(f"saved raw bundle: {save_path} shape={features.shape}")


def preprocess_to_mtl(split_name: str, bundle_name: str) -> None:
    feature_path = DATA_INTERIM / "inputfeatures" / bundle_name
    features = np.load(feature_path / "numpy_features.npy")
    labels = np.load(feature_path / "numpy_labels.npy")
    lats = np.load(feature_path / "numpy_lat.npy")
    lons = np.load(feature_path / "numpy_lon.npy")
    x_mean = np.load(feature_path / "features_mean.npy").astype(float)
    x_std = np.load(feature_path / "features_std.npy").astype(float)

    features, labels, lats, lons = preprocess_features_labels(features, labels, lats, lons, x_mean, x_std)
    out_dir = DATA_INTERIM / "toc_preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"features_{split_name}_toc.npy", features)
    np.save(out_dir / f"labels_{split_name}_toc.npy", labels)
    np.save(out_dir / f"lats_{split_name}_toc.npy", lats)
    np.save(out_dir / f"lons_{split_name}_toc.npy", lons)
    print(f"preprocessed {split_name}: {features.shape}")


def main() -> None:
    set_seed(42)
    build_bundle("toc_deep_v2.csv", "SedTOCFeaturesnoNAN_TOC_DO_men")
    build_bundle("toc_continentalshelves_v2.csv", "SedTOCFeaturesnoNAN_TOC_CS_men")
    preprocess_to_mtl("DO", "SedTOCFeaturesnoNAN_TOC_DO_men")
    preprocess_to_mtl("CS", "SedTOCFeaturesnoNAN_TOC_CS_men")


if __name__ == "__main__":
    main()
