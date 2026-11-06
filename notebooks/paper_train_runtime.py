"""Paper-faithful train / eval / chunked-map helpers for notebooks.

Uses the nn4sedratetoc stack that produced Table 2:
  masked_features (values_only / attention), trainDNN / trainMTL, Huber, seed=42.

Notebooks should call these helpers so training, testing, and per-chunk world
prediction actually run (not load cached paper figures).
"""
from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

MTL_ROOT = Path("/gxfs_work/cau/sunms498/work/mtl-mar")
NN4_ROOT = Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc")
NN4_NB = NN4_ROOT / "notebooks" / "SedRate"
NN4_MODELS = NN4_ROOT / "src" / "models"
NN4_POST = NN4_ROOT / "src" / "postprocessing"

TABLE2 = {
    "random_forest": {"test_mae": 19.50, "test_cc": 0.93, "budget": 68.27},
    "xgboost": {"test_mae": 24.53, "test_cc": 0.90, "budget": 65.46},
    "catboost": {"test_mae": 34.25, "test_cc": 0.90, "budget": 66.12},
    "knn": {"test_mae": 19.93, "test_cc": 0.92, "budget": 50.60},
    "tabpfn": {"test_mae": 17.27, "test_cc": 0.93, "budget": 40.39},
    "ffnn": {"test_mae": 20.18, "test_cc": 0.91, "budget": 34.82},
    "mtl": {"test_mae": 18.90, "test_cc": 0.92, "budget": 28.9},
    "ffnn_attention": {"test_mae": 20.50, "test_cc": 0.91, "budget": 33.38},
    "mtl_attention": {"test_mae": 20.29, "test_cc": 0.91, "budget": 36.77},
}

# Prediction .npy grids that are south-up relative to island_map
SOUTH_UP_WHEN_SAVED_WITHOUT_ROT = frozenset(
    {"random_forest", "xgboost", "catboost", "knn", "ffnn", "ffnn_attention"}
)


def setup_nn4_path(variant: str = "values_only") -> None:
    os.environ["MASKED_FEATURE_VARIANT"] = variant
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    for p in (str(NN4_NB), str(NN4_MODELS), str(NN4_POST)):
        if p not in sys.path:
            sys.path.insert(0, p)


def set_deterministic(seed: int = 42) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compare_table2(model_key: str, test_mae: float, test_cc: float) -> None:
    t = TABLE2[model_key]
    print(
        f"Table 2 target ({model_key}): test MAE={t['test_mae']:.2f}, CC={t['test_cc']:.2f}, "
        f"budget={t['budget']}\n"
        f"This run:                 test MAE={test_mae:.4f}, CC={test_cc:.4f}\n"
        f"delta MAE={test_mae - t['test_mae']:+.4f}, delta CC={test_cc - t['test_cc']:+.4f}"
    )


def plot_scatter(y_train, p_train, y_test, p_test, title: str, save_path: Path | None = None):
    tr, te = metrics(y_train, p_train), metrics(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_train, p_train, s=8, alpha=0.35, label=f"train MAE={tr['mae']:.2f}, r={tr['corr']:.3f}")
    ax.scatter(y_test, p_test, s=8, alpha=0.45, label=f"test MAE={te['mae']:.2f}, r={te['corr']:.3f}")
    lims = [
        min(np.min(y_train), np.min(y_test), np.min(p_train), np.min(p_test)),
        max(np.max(y_train), np.max(y_test), np.max(p_train), np.max(p_test)),
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("True MAR")
    ax.set_ylabel("Predicted MAR")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("Saved scatter:", save_path)
    plt.show()
    plt.close(fig)
    return tr, te


def plot_learning_curves(train_mae, eval_mae, title: str, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(train_mae, label="train MAE")
    ax.plot(eval_mae, label="test MAE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MAE")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def load_land() -> np.ndarray:
    land = np.load(MTL_ROOT / "data/raw/island_map.npy")
    return np.where(np.isnan(land), 1, land).astype(bool)


def save_and_plot_mar_map(
    prediction_map: np.ndarray,
    out_npy: Path,
    out_png: Path,
    title: str,
    *,
    already_north_up: bool = True,
) -> np.ndarray:
    """Save map (north-up, land zeroed) and paper-style batlow Robinson PNG."""
    from paper_results_display import plot_mar_robinson_png

    arr = np.asarray(prediction_map, dtype=float).copy()
    if not already_north_up:
        arr = np.flipud(arr)
    land = load_land()
    arr = arr.copy()
    arr[land] = 0.0
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, arr)
    print("Saved map npy:", out_npy)
    png = plot_mar_robinson_png(arr, land, title=title, project_root=MTL_ROOT, dpi=120)
    out_png.write_bytes(png)
    print("Saved map png:", out_png)
    # also display if in notebook
    try:
        from IPython.display import Image, display

        display(Image(data=png))
    except Exception:
        pass
    return arr


def load_mar_values_only(test_size: float = 0.25, seed: int = 42):
    """Paper values_only MAR split (113-d)."""
    setup_nn4_path("values_only")
    from masked_features import (  # noqa: WPS433
        load_filtered_mask_stack,
        preprocess_features_labels_masked,
        variant_data_path,
    )
    from trainDNN import set_seed, split_train_test_data  # noqa: WPS433

    set_seed(seed)
    mask_stack = load_filtered_mask_stack()
    parts = []
    for basin in ("DO", "CS"):
        path = variant_data_path("MAR", basin)
        print(f"Loading MAR {basin}: {path}")
        feats, labs, *_rest = preprocess_features_labels_masked(str(path), mask_stack=mask_stack)
        y_binned = pd.qcut(labs.reshape(-1), q=10, labels=False, duplicates="drop")
        tr_x, tr_y, te_x, te_y = split_train_test_data(
            feats, labs, test_size_ratio=test_size, random_seed=seed, stratify=y_binned
        )
        parts.append((tr_x, tr_y, te_x, te_y, feats, labs))
    train_x = np.vstack([p[0] for p in parts])
    train_y = np.concatenate([p[1].reshape(-1) for p in parts])
    test_x = np.vstack([p[2] for p in parts])
    test_y = np.concatenate([p[3].reshape(-1) for p in parts])
    print(f"values_only: train={train_x.shape} test={test_x.shape}")
    return {
        "train_features": train_x,
        "train_labels": train_y,
        "test_features": test_x,
        "test_labels": test_y,
        "n_features": train_x.shape[1],
    }


def load_mar_attention(test_size: float = 0.25, seed: int = 42):
    """Paper attention MAR split (226-d = values + masks)."""
    setup_nn4_path("attention")
    from masked_features import (  # noqa: WPS433
        load_filtered_mask_stack,
        preprocess_features_labels_masked,
        variant_data_path,
    )
    from trainDNN import set_seed, split_train_test_data  # noqa: WPS433

    set_seed(seed)
    # attention uses imputed+mask channels; variant path still *_men_masked
    os.environ["MASKED_FEATURE_VARIANT"] = "imputed"
    mask_stack = load_filtered_mask_stack()
    parts = []
    for basin in ("DO", "CS"):
        path = variant_data_path("MAR", basin)
        print(f"Loading MAR attention {basin}: {path}")
        feats, labs, *_rest = preprocess_features_labels_masked(str(path), mask_stack=mask_stack)
        y_binned = pd.qcut(labs.reshape(-1), q=10, labels=False, duplicates="drop")
        tr_x, tr_y, te_x, te_y = split_train_test_data(
            feats, labs, test_size_ratio=test_size, random_seed=seed, stratify=y_binned
        )
        parts.append((tr_x, tr_y, te_x, te_y))
    train_x = np.vstack([p[0] for p in parts])
    train_y = np.concatenate([p[1].reshape(-1) for p in parts])
    test_x = np.vstack([p[2] for p in parts])
    test_y = np.concatenate([p[3].reshape(-1) for p in parts])
    assert train_x.shape[1] == 226, train_x.shape
    print(f"attention: train={train_x.shape} test={test_x.shape}")
    return {
        "train_features": train_x,
        "train_labels": train_y,
        "test_features": test_x,
        "test_labels": test_y,
        "n_features": 226,
        "n_value_features": 113,
    }


def make_loaders(train_x, train_y, test_x, test_y, batch_size: int = 64):
    train_ds = TensorDataset(
        torch.tensor(train_x, dtype=torch.float64),
        torch.tensor(train_y, dtype=torch.float64),
    )
    test_ds = TensorDataset(
        torch.tensor(test_x, dtype=torch.float64),
        torch.tensor(test_y, dtype=torch.float64),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


def predict_loader(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float64)
            out = model(x)
            ys.append(y.numpy().reshape(-1))
            ps.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


def world_feature_dirs_attention():
    """Chunk dirs used for attention (imputed+mask) world prediction."""
    # Prefer nn4 interim world chunks
    candidates = [
        NN4_ROOT / "data/interim/features_men_masked/world",
        NN4_ROOT / "data/interim/features/features_men_masked",
        MTL_ROOT / "data/interim/features_men_masked/world",
    ]
    for c in candidates:
        if c.exists():
            files = sorted(c.glob("*.npy"))
            if files:
                return c, files
    raise FileNotFoundError("No attention world feature chunks found")


def predict_world_chunks_attention(model, device, x_mean, x_std, chunk_shape=(6, 4320)):
    """Predict MAR for each world chunk; print progress (paper attention path)."""
    setup_nn4_path("imputed")
    from masked_features import normalize_and_apply_feature_mask, load_filtered_mask_stack  # noqa: WPS433

    world_dir, files = world_feature_dirs_attention()
    print(f"World chunks: {len(files)} in {world_dir}")
    mask_stack = load_filtered_mask_stack()
    prediction_map = None
    model.eval()
    with torch.no_grad():
        for i, f in enumerate(files):
            raw = np.load(f)
            # raw chunks are often (n_cells, n_feat) or (6,4320,n)
            if raw.ndim == 3:
                feats = raw.reshape(-1, raw.shape[-1])
            else:
                feats = raw
            feats = normalize_and_apply_feature_mask(feats, x_mean, x_std, mask_stack=mask_stack)
            t = torch.tensor(feats, dtype=torch.float64, device=device)
            preds = model(t).detach().cpu().numpy().reshape(chunk_shape)
            prediction_map = preds if prediction_map is None else np.append(prediction_map, preds, axis=0)
            if (i + 1) % 20 == 0 or (i + 1) == len(files):
                print(f"  predicted chunk {i+1}/{len(files)}")
            del t, feats, raw
            gc.collect()
    # Match nn4 postprocess orientation used when saving supervised maps
    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))
    return prediction_map
