"""Train global MAR tree/KNN models with 111 features (final configs from notebook)."""

from __future__ import annotations

import gc
import json
import sys
import warnings
from os import walk
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.extend([str(project_root / "src" / "models"), str(project_root / "src" / "postprocessing")])

from utils import DATA_INTERIM, DATA_OUTPUT, DATA_RAW  # noqa: E402
from train_neural_network import preprocess_features_labels, set_seed, split_train_test_data  # noqa: E402
from tree_based_models import train_tree_based_model  # noqa: E402


def load_mar_split(region: str):
    feature_path = DATA_INTERIM / "inputfeatures" / f"SedTOCFeaturesnoNAN_MAR_{region}_men"
    features = np.load(feature_path / "numpy_features.npy")
    labels = np.load(feature_path / "numpy_labels.npy")
    x_mean = np.load(feature_path / "features_mean.npy").astype(float)
    x_std = np.load(feature_path / "features_std.npy").astype(float)
    lats = np.load(feature_path / "numpy_lat.npy")
    lons = np.load(feature_path / "numpy_lon.npy")
    features, labels, lats, lons = preprocess_features_labels(features, labels, lats, lons, x_mean, x_std)
    y_binned = pd.qcut(labels, q=10, labels=False, duplicates="drop")
    train_x, train_y, test_x, test_y = split_train_test_data(
        features, labels, test_size_ratio=0.25, random_seed=42, stratify=y_binned
    )
    return train_x, train_y, test_x, test_y, x_mean, x_std


def metrics(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "corr": float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]),
    }


def predict_global_map(model, x_mean, x_std, out_name: str) -> None:
    dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
    files = sorted(f for f in dataset_path.iterdir() if f.is_file() and f.suffix == ".npy")
    chunk_shape = (6, 4320)
    first_run = True
    prediction_map = None

    for file in files:
        features = np.load(file)
        features = np.divide((features - x_mean), x_std)
        preds = model.predict(features).reshape(chunk_shape)
        if first_run:
            prediction_map = preds
            first_run = False
        else:
            prediction_map = np.append(prediction_map, preds, axis=0)

    land_map = np.load(DATA_RAW / "island_map.npy")
    land_map[np.isnan(land_map)] = 1
    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))
    prediction_map[land_map] = 0

    out_dir = DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / out_name, prediction_map)
    print(f"saved map: {out_dir / out_name}")


def main() -> None:
    set_seed(42)
    world_dim = np.load(
        sorted((DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men").glob("*.npy"))[0]
    ).shape[1]
    print(f"world feature dim: {world_dim}")

    train_do, y_train_do, test_do, y_test_do, _, _ = load_mar_split("DO")
    train_cs, y_train_cs, test_cs, y_test_cs, x_mean, x_std = load_mar_split("CS")
    assert train_do.shape[1] == world_dim == train_cs.shape[1]

    train_x = np.concatenate([train_cs, train_do], axis=0)
    train_y = np.concatenate([y_train_cs, y_train_do], axis=0)
    test_x = np.concatenate([test_cs, test_do], axis=0)
    test_y = np.concatenate([y_test_cs, y_test_do], axis=0)
    print(f"train/test: {train_x.shape} / {test_x.shape}, n_features={train_x.shape[1]}")

    configs = [
        ("random_forest", {"model_type": "random_forest", "n_estimators": 60}),
        ("xgboost", {"model_type": "xgboost", "n_estimators": 1000}),
        ("catboost", {"model_type": "catboost", "n_estimators": 1000}),
    ]
    results = {}

    for name, kwargs in configs:
        print(f"\nTraining {name}...")
        model, _, _, train_pred, test_pred = train_tree_based_model(train_x, train_y, test_x, test_y, **kwargs)
        results[name] = {"train": metrics(train_y, train_pred), "test": metrics(test_y, test_pred)}
        print(results[name])
        predict_global_map(model, x_mean, x_std, f"prediction_map_MAR_entire_{name}_111feat.npy")
        gc.collect()

    print("\nTraining KNN k=9...")
    knn = KNeighborsRegressor(n_neighbors=9, weights="uniform", metric="minkowski", p=2)
    knn.fit(train_x, train_y)
    train_pred = knn.predict(train_x)
    test_pred = knn.predict(test_x)
    results["knn"] = {"train": metrics(train_y, train_pred), "test": metrics(test_y, test_pred)}
    print(results["knn"])
    predict_global_map(knn, x_mean, x_std, "prediction_map_MAR_entire_knn_111feat.npy")

    metrics_path = DATA_OUTPUT / "model_metrics" / "tree_models_111feat.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nmetrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
