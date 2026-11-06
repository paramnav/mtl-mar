"""Train one global MTL model (111 features) for MAR + TOC."""

from __future__ import annotations

import gc
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / "src"))
sys.path.extend([str(project_root / "src" / "models"), str(project_root / "src" / "postprocessing")])

from utils import DATA_INTERIM, DATA_OUTPUT, DATA_RAW  # noqa: E402
from train_multi_task_learning import (  # noqa: E402
    buildDNN,
    set_seed,
    split_train_test_data,
    train_model,
)
from train_neural_network import preprocess_features_labels  # noqa: E402


class CombinedLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = {
            "sedimentation_rates": torch.nn.HuberLoss(),
            "toc": torch.nn.HuberLoss(),
        }

    def forward(self, features, labels, targets, decoded, porosity, train_step, task_id):
        del features, decoded, porosity
        labels = labels.reshape(labels.shape[0], 1)
        targets = targets.reshape(targets.shape[0], 1)
        if task_id == "sedimentation_rates":
            return self.loss_fn["sedimentation_rates"](targets, labels)
        return self.loss_fn["toc"](targets, labels)


def load_toc_split(region: str):
    toc_path = DATA_INTERIM / "toc_preprocessed"
    features = np.load(toc_path / f"features_{region}_toc.npy")
    labels = np.load(toc_path / f"labels_{region}_toc.npy")
    y_binned = pd.qcut(labels, q=10, labels=False, duplicates="drop")
    return split_train_test_data(features, labels, test_size_ratio=0.25, random_seed=42, stratify=y_binned)


def load_mar_split(region: str):
    feature_path = DATA_INTERIM / "inputfeatures" / f"SedTOCFeaturesnoNAN_MAR_{region}_men"
    features = np.load(feature_path / "numpy_features.npy")
    labels = np.load(feature_path / "numpy_labels.npy")
    lats = np.load(feature_path / "numpy_lat.npy")
    lons = np.load(feature_path / "numpy_lon.npy")
    x_mean = np.load(feature_path / "features_mean.npy").astype(float)
    x_std = np.load(feature_path / "features_std.npy").astype(float)
    features, labels, _, _ = preprocess_features_labels(features, labels, lats, lons, x_mean, x_std)
    y_binned = pd.qcut(labels, q=10, labels=False, duplicates="drop")
    return split_train_test_data(features, labels, test_size_ratio=0.25, random_seed=42, stratify=y_binned)


def eval_task(model, features, labels, task_id, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float64, device=device)
        _, preds = model(x, task_id)
        pred = preds.cpu().numpy().ravel()
    return {
        "mae": float(mean_absolute_error(labels, pred)),
        "r2": float(r2_score(labels, pred)),
        "corr": float(np.corrcoef(labels, pred)[0, 1]),
    }


def predict_global_maps(model, x_mean, x_std, device, count_tag: str) -> None:
    dataset_path = DATA_INTERIM / "worldfeatures" / "WorldFeaturesAllnoNaN_SedTOC_predict_men"
    files = sorted(f for f in dataset_path.iterdir() if f.suffix == ".npy")
    chunk_shape = (6, 4320)
    mar_map = None
    toc_map = None

    model.eval()
    with torch.no_grad():
        for file in files:
            features = np.load(file)
            features = np.divide((features - x_mean), x_std)
            x = torch.tensor(features, dtype=torch.float64, device=device)
            _, mar_pred = model(x, "sedimentation_rates")
            _, toc_pred = model(x, "toc")
            mar_chunk = mar_pred.cpu().numpy().reshape(chunk_shape)
            toc_chunk = toc_pred.cpu().numpy().reshape(chunk_shape)
            mar_map = mar_chunk if mar_map is None else np.append(mar_map, mar_chunk, axis=0)
            toc_map = toc_chunk if toc_map is None else np.append(toc_map, toc_chunk, axis=0)

    land_map = np.load(DATA_RAW / "island_map.npy")
    land_map[np.isnan(land_map)] = 1
    mar_map = np.rot90(np.rot90(np.fliplr(mar_map)))
    toc_map = np.rot90(np.rot90(np.fliplr(toc_map)))
    mar_map[land_map] = 0
    toc_map[land_map] = 0

    out_dir = DATA_OUTPUT / "predictionmaps_SedRate" / "hyperparameter_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"prediction_map_MAR_entire_MTL_111feat_{count_tag}.npy", mar_map)
    np.save(out_dir / f"prediction_map_toc_entire_MTL_111feat_{count_tag}.npy", toc_map)


def main() -> None:
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    toc_train_cs, toc_y_train_cs, toc_test_cs, toc_y_test_cs = load_toc_split("CS")
    toc_train_do, toc_y_train_do, toc_test_do, toc_y_test_do = load_toc_split("DO")
    mar_train_cs, mar_y_train_cs, mar_test_cs, mar_y_test_cs = load_mar_split("CS")
    mar_train_do, mar_y_train_do, mar_test_do, mar_y_test_do = load_mar_split("DO")

    toc_train_x = np.concatenate([toc_train_cs, toc_train_do])
    toc_train_y = np.concatenate([toc_y_train_cs, toc_y_train_do])
    toc_test_x = np.concatenate([toc_test_cs, toc_test_do])
    toc_test_y = np.concatenate([toc_y_test_cs, toc_y_test_do])

    mar_train_x = np.concatenate([mar_train_cs, mar_train_do])
    mar_train_y = np.concatenate([mar_y_train_cs, mar_y_train_do])
    mar_test_x = np.concatenate([mar_test_cs, mar_test_do])
    mar_test_y = np.concatenate([mar_y_test_cs, mar_y_test_do])

    assert toc_train_x.shape[1] == mar_train_x.shape[1]
    print(f"feature dim={mar_train_x.shape[1]}")
    print(f"MAR train/test: {mar_train_x.shape}/{mar_test_x.shape}")
    print(f"TOC train/test: {toc_train_x.shape}/{toc_test_x.shape}")

    layer_widths = [512, 256, 128, 64]
    num_shared_layers = 3
    num_epochs = 500 if device.type == "cpu" else 2000

    train_loaders = {
        "sedimentation_rates": DataLoader(
            TensorDataset(torch.tensor(mar_train_x, dtype=torch.float32), torch.tensor(mar_train_y, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        ),
        "toc": DataLoader(
            TensorDataset(torch.tensor(toc_train_x, dtype=torch.float32), torch.tensor(toc_train_y, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        ),
    }
    test_loaders = {
        "sedimentation_rates": DataLoader(
            TensorDataset(torch.tensor(mar_test_x, dtype=torch.float32), torch.tensor(mar_test_y, dtype=torch.float32)),
            batch_size=32,
            shuffle=False,
        ),
        "toc": DataLoader(
            TensorDataset(torch.tensor(toc_test_x, dtype=torch.float32), torch.tensor(toc_test_y, dtype=torch.float32)),
            batch_size=32,
            shuffle=False,
        ),
    }

    model = buildDNN(layer_widths, mar_train_x, num_shared_layers).to(device)
    loss_fn = CombinedLoss().to(device)
    print(f"training MTL: layers={layer_widths}, shared={num_shared_layers}, epochs={num_epochs}")
    train_model(model, loss_fn, train_loaders, test_loaders, train_step=2, num_epochs=num_epochs)

    results = {
        "mar_train": eval_task(model, mar_train_x, mar_train_y, "sedimentation_rates", device),
        "mar_test": eval_task(model, mar_test_x, mar_y_test, "sedimentation_rates", device),
        "toc_train": eval_task(model, toc_train_x, toc_train_y, "toc", device),
        "toc_test": eval_task(model, toc_test_x, toc_test_y, "toc", device),
        "layer_widths": layer_widths,
        "num_shared_layers": num_shared_layers,
        "num_epochs": num_epochs,
        "n_features": int(mar_train_x.shape[1]),
    }
    print(json.dumps(results, indent=2))

    x_mean = np.load(DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men" / "features_mean.npy").astype(float)
    x_std = np.load(DATA_INTERIM / "inputfeatures" / "SedTOCFeaturesnoNAN_MAR_DO_men" / "features_std.npy").astype(float)
    predict_global_maps(model, x_mean, x_std, device, "0")

    out_path = DATA_OUTPUT / "model_metrics" / "mtl_111feat.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    gc.collect()


if __name__ == "__main__":
    main()
