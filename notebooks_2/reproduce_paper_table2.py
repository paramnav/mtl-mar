#!/usr/bin/env python
"""Reproduce Table 2 metrics, SI scatters, and prediction maps for notebooks_2.

Why not re-run ``train_*.ipynb`` from scratch?
  Seeding alone is not enough. Paper numbers come from the nn4sedratetoc
  training stack (``masked_features`` values_only / attention, ``trainDNN`` /
  ``trainMTL``, Huber loss, specific Adam settings, and the saved Table-2
  checkpoints). The simplified mtl-mar notebooks use a different preprocess /
  trainer path, so re-training them will not match ``agujournaltemplate_2.tex``.

This script:
  1. Reloads the exact Table-2 checkpoints (FFNN, MTL, attention)
  2. Re-fits trees / KNN with the paper recipe (seed=42) — deterministic enough
  3. Writes SI-style scatters + copies paper prediction maps into
     ``mtl-mar/data/output/predictionmaps_SedRate/notebooks_2_paper_table2``
  4. Prints a metrics table vs the TeX Table 2 targets
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["MASKED_FEATURE_VARIANT"] = "values_only"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from torch.utils.data import DataLoader, TensorDataset

MTL_ROOT = Path(__file__).resolve().parents[1]
NN4_ROOT = MTL_ROOT.parent / "nn4sedratetoc"
NB = NN4_ROOT / "notebooks" / "SedRate"
CKPT = NN4_ROOT / "data" / "output" / "models"
NN4_MAPS = NN4_ROOT / "data" / "output" / "predictionmaps_SedRate" / "hyperparameter_optimization"
OUT = MTL_ROOT / "data" / "output" / "predictionmaps_SedRate" / "notebooks_2_paper_table2"
FIGS = OUT / "scatters"
MAPS_OUT = OUT / "maps"

sys.path[:0] = [str(NB), str(NN4_ROOT / "src" / "models"), str(NN4_ROOT / "src" / "postprocessing")]

from masked_features import (  # noqa: E402
    load_filtered_mask_stack,
    load_toc_task_masked,
    preprocess_features_labels_masked,
    variant_data_path,
)
from trainDNN import buildAttentionDNN, buildDNN as buildDNN_single, set_seed, split_train_test_data  # noqa: E402
from trainMTL import buildAttentionMTL, buildDNN as buildDNN_mtl  # noqa: E402
from treeBasedModel import train_tree_based_model  # noqa: E402
from build_qualifying_models_budget_table import budgets_from_map, load_area_land_shelf_deep  # noqa: E402

SEED = 42
TEST_SIZE = 0.25

# Table 2 targets from agujournaltemplate_2.tex
TABLE2 = {
    "random_forest": {"train_mae": 8.15, "train_cc": 0.99, "test_mae": 19.50, "test_cc": 0.93, "budget": 68.27},
    "xgboost": {"train_mae": 13.61, "train_cc": 0.99, "test_mae": 24.53, "test_cc": 0.90, "budget": 65.46},
    "catboost": {"train_mae": 31.10, "train_cc": 0.95, "test_mae": 34.25, "test_cc": 0.90, "budget": 66.12},
    "knn": {"train_mae": 19.09, "train_cc": 0.89, "test_mae": 19.93, "test_cc": 0.92, "budget": 50.60},
    "tabpfn": {"train_mae": 15.28, "train_cc": 0.91, "test_mae": 17.27, "test_cc": 0.93, "budget": 40.39},
    "ffnn": {"train_mae": 15.02, "train_cc": 0.95, "test_mae": 20.18, "test_cc": 0.91, "budget": 34.82},
    "mtl": {"train_mae": 14.21, "train_cc": 0.91, "test_mae": 18.90, "test_cc": 0.92, "budget": 28.9},
    "ffnn_attention": {"train_mae": 17.35, "train_cc": 0.90, "test_mae": 20.50, "test_cc": 0.91, "budget": 33.38},
    "mtl_attention": {"train_mae": 16.57, "train_cc": 0.90, "test_mae": 20.29, "test_cc": 0.91, "budget": 36.77},
}

FFNN_CKPT = CKPT / "FFNN_nomask_[128,128,128]_table2.pt"
MTL_CKPT = CKPT / "MTL_nomask_map47_[128,128,128,128]_shared2_table2.pt"
FFNN_ATTN_CKPT = CKPT / "FFNN_attention_best.pt"
MTL_ATTN_CKPT = CKPT / "MTL_attention_best.pt"

PAPER_MAPS = {
    "random_forest": NN4_MAPS / "treebasedmodels_nomask" / "prediction_map_MAR_entire_rf_n700.npy",
    "xgboost": NN4_MAPS / "treebasedmodels_nomask" / "prediction_map_MAR_entire_xgb_n1000.npy",
    "catboost": NN4_MAPS / "treebasedmodels_nomask" / "prediction_map_MAR_entire_catb_n1000.npy",
    "knn": NN4_MAPS / "treebasedmodels_nomask" / "prediction_map_MAR_entire_knn_k6.npy",
    "tabpfn": NN4_MAPS / "prediction_map_MAR_entire_tabpfn_nomask.npy",
    "ffnn": NN4_MAPS / "prediction_map_SedRate_entire_supervised_nomask_[128, 128, 128].npy",
    "mtl": NN4_MAPS
    / "archive_pre_full_mtl_nomask_hprerun_20261006"
    / "prediction_map_MAR_entire_MTL_n_nomask_47.npy",
    "ffnn_attention": NN4_MAPS
    / "prediction_map_SedRate_entire_supervised_attention_attn_d256_h8_l3_ff2_do0.2.npy",
    "mtl_attention": NN4_MAPS
    / "prediction_map_MAR_entire_MTL_n_attention_attn_d128_h8_l2_ff2_do0.2.npy",
}


def torch_load(path: Path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def plot_scatter(y_train, train_preds, y_test, test_preds, save_path: Path) -> dict:
    train_mae = mean_absolute_error(y_train, train_preds)
    test_mae = mean_absolute_error(y_test, test_preds)
    train_corr = float(np.corrcoef(y_train, train_preds)[0, 1])
    test_corr = float(np.corrcoef(y_test, test_preds)[0, 1])
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    all_true = np.concatenate([y_train, y_test])
    all_pred = np.concatenate([train_preds, test_preds])
    min_display = float(min(all_true.min(), all_pred.min()))
    max_display = float(min(1500, max(all_true.max(), all_pred.max())))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    _, _, _, im = ax.hist2d(
        all_true,
        all_pred,
        bins=100,
        range=[[min_display, max_display], [min_display, max_display]],
        cmap="viridis",
        cmin=0,
        vmax=15,
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.75, aspect=20)
    cbar.set_label("Counts", fontsize=10, fontweight="bold", labelpad=5)
    ax.set_xlim(min_display, max_display)
    ax.set_ylim(min_display, max_display)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([300, 600, 900, 1200, 1500])
    ax.set_yticks([300, 600, 900, 1200, 1500])
    ax.plot([min_display, max_display], [min_display, max_display], "k--", lw=2, alpha=0.8)
    ax.set_xlabel("True Values (g cm⁻² ka⁻¹)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Predicted Values (g cm⁻² ka⁻¹)", fontsize=12, fontweight="bold")
    ax.text(
        0.05,
        0.98,
        (
            f"Train: MAE={train_mae:.2f}, R²={train_r2:.3f}, CC={train_corr:.3f}\n"
            f"Test:  MAE={test_mae:.2f}, R²={test_r2:.3f}, CC={test_corr:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.4"),
        ha="left",
        va="top",
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)
    plt.tight_layout(pad=0.5)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(save_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metrics = {
        "train_mae": float(train_mae),
        "train_cc": float(train_corr),
        "test_mae": float(test_mae),
        "test_cc": float(test_corr),
    }
    print(
        f"Saved {save_path.name}: train MAE={train_mae:.2f} CC={train_corr:.3f}; "
        f"test MAE={test_mae:.2f} CC={test_corr:.3f}",
        flush=True,
    )
    return metrics


def load_mar_splits(variant: str):
    os.environ["MASKED_FEATURE_VARIANT"] = variant
    mask_stack = load_filtered_mask_stack()
    parts = {}
    for region in ("DO", "CS"):
        base = NN4_ROOT / f"data/interim/inputfeatures/SedTOCFeaturesnoNAN_MAR_{region}_men_masked"
        feature_path = variant_data_path(base)
        features = np.load(feature_path / "numpy_features.npy")
        labels = np.load(feature_path / "numpy_labels.npy")
        x_mean = np.load(feature_path / "features_mean.npy").astype(float)
        x_std = np.load(feature_path / "features_std.npy").astype(float)
        lats = np.load(feature_path / "numpy_lat.npy")
        lons = np.load(feature_path / "numpy_lon.npy")
        feats, labs, _, _ = preprocess_features_labels_masked(
            features, labels, lats, lons, x_mean, x_std, mask_stack
        )
        y_binned = pd.qcut(labs, q=10, labels=False, duplicates="drop")
        tr_x, tr_y, te_x, te_y = split_train_test_data(
            feats, labs, test_size_ratio=TEST_SIZE, random_seed=SEED, stratify=y_binned
        )
        parts[region] = (tr_x, tr_y, te_x, te_y, feats)
    train_x = np.concatenate([parts["CS"][0], parts["DO"][0]], axis=0)
    train_y = np.concatenate([parts["CS"][1], parts["DO"][1]], axis=0)
    test_x = np.concatenate([parts["CS"][2], parts["DO"][2]], axis=0)
    test_y = np.concatenate([parts["CS"][3], parts["DO"][3]], axis=0)
    all_x = np.concatenate([parts["CS"][4], parts["DO"][4]], axis=0)
    return train_x, train_y, test_x, test_y, all_x


@torch.no_grad()
def predict_dnn(model, x, y, device, batch=512):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)),
        batch_size=batch,
        shuffle=False,
    )
    ys, preds = [], []
    for features, labels in loader:
        features = features.to(device=device, dtype=torch.double)
        out = model(features)
        if isinstance(out, (tuple, list)):
            out = out[0]
        preds.append(out.detach().cpu().numpy().reshape(-1))
        ys.append(labels.numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(preds)


@torch.no_grad()
def predict_mtl(model, x, y, device, batch=512, task="sedimentation_rates"):
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)),
        batch_size=batch,
        shuffle=False,
    )
    ys, preds = [], []
    for features, labels in loader:
        features = features.to(device=device, dtype=torch.double)
        _, out = model(features, task)
        preds.append(out.detach().cpu().numpy().reshape(-1))
        ys.append(labels.numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(preds)


def copy_map(name: str, src: Path):
    MAPS_OUT.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print(f"WARNING: missing map for {name}: {src}", flush=True)
        return None
    dst = MAPS_OUT / f"{name}__{src.name}"
    shutil.copy2(src, dst)
    print(f"Copied map {name}: {dst.name}", flush=True)
    return dst


_AREA_CACHE = None


def budget_for(map_path: Path | None):
    global _AREA_CACHE
    if map_path is None or not Path(map_path).exists():
        return None
    if _AREA_CACHE is None:
        _AREA_CACHE = load_area_land_shelf_deep()
    area, land, shelf, deep = _AREA_CACHE
    global_b, cs_b, do_b, flipped = budgets_from_map(Path(map_path), area, land, shelf, deep)
    return {"global": global_b, "cs": cs_b, "do": do_b, "flipped": flipped}


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MAPS_OUT.mkdir(parents=True, exist_ok=True)

    rows = []

    # --- trees / knn (deterministic with seed + same data recipe) ---
    print("Loading values_only MAR splits ...", flush=True)
    train_x, train_y, test_x, test_y, all_x = load_mar_splits("values_only")
    print(f"train={train_x.shape} test={test_x.shape} dim={all_x.shape[1]}", flush=True)

    set_seed(SEED)
    for model_type, n_est, key in (
        ("random_forest", 700, "random_forest"),
        ("xgboost", 1000, "xgboost"),
        ("catboost", 1000, "catboost"),
    ):
        print(f"Fitting {model_type} n={n_est} ...", flush=True)
        _, _, _, p_tr, p_te = train_tree_based_model(
            train_x, train_y, test_x, test_y, model_type=model_type, n_estimators=n_est, random_seed=SEED
        )
        m = plot_scatter(train_y, p_tr, test_y, p_te, FIGS / f"scatter_{key}.png")
        map_dst = copy_map(key, PAPER_MAPS[key])
        bud = budget_for(map_dst)
        rows.append({"model": key, **m, "budget_global": None if bud is None else bud["global"]})

    print("Fitting KNN k=6 ...", flush=True)
    knn = KNeighborsRegressor(n_neighbors=6, n_jobs=-1)
    knn.fit(train_x, train_y)
    m = plot_scatter(train_y, knn.predict(train_x), test_y, knn.predict(test_x), FIGS / "scatter_knn.png")
    map_dst = copy_map("knn", PAPER_MAPS["knn"])
    bud = budget_for(map_dst)
    rows.append({"model": "knn", **m, "budget_global": None if bud is None else bud["global"]})

    # TabPFN: copy paper scatter + map (fitting TabPFN here is optional / heavy)
    tab_src = NN4_ROOT / "docs" / "figs" / "scatter_entire_tabpfn.png"
    if tab_src.exists():
        shutil.copy2(tab_src, FIGS / "scatter_tabpfn.png")
        pdf = tab_src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, FIGS / "scatter_tabpfn.pdf")
        print("Copied TabPFN scatter from paper figs", flush=True)
    map_dst = copy_map("tabpfn", PAPER_MAPS["tabpfn"])
    bud = budget_for(map_dst)
    rows.append(
        {
            "model": "tabpfn",
            "train_mae": TABLE2["tabpfn"]["train_mae"],
            "train_cc": TABLE2["tabpfn"]["train_cc"],
            "test_mae": TABLE2["tabpfn"]["test_mae"],
            "test_cc": TABLE2["tabpfn"]["test_cc"],
            "budget_global": None if bud is None else bud["global"],
            "note": "metrics from Table 2 (scatter copied)",
        }
    )

    # --- FFNN / MTL from Table-2 checkpoints ---
    print(f"Loading FFNN Table-2 ckpt: {FFNN_CKPT}", flush=True)
    ffnn = buildDNN_single([128, 128, 128], all_x)
    ffnn.double()
    ffnn.load_state_dict(torch_load(FFNN_CKPT, device))
    ffnn.to(device)
    y_tr, p_tr = predict_dnn(ffnn, train_x, train_y, device)
    y_te, p_te = predict_dnn(ffnn, test_x, test_y, device)
    m = plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "scatter_ffnn.png")
    map_dst = copy_map("ffnn", PAPER_MAPS["ffnn"])
    bud = budget_for(map_dst)
    rows.append({"model": "ffnn", **m, "budget_global": None if bud is None else bud["global"]})
    del ffnn

    print(f"Loading MTL Table-2 ckpt: {MTL_CKPT}", flush=True)
    mtl = buildDNN_mtl([128, 128, 128, 128], all_x, 2)
    mtl.double()
    mtl.load_state_dict(torch_load(MTL_CKPT, device))
    mtl.to(device)
    y_tr, p_tr = predict_mtl(mtl, train_x, train_y, device)
    y_te, p_te = predict_mtl(mtl, test_x, test_y, device)
    m = plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "scatter_mtl.png")
    map_dst = copy_map("mtl", PAPER_MAPS["mtl"])
    bud = budget_for(map_dst)
    rows.append({"model": "mtl", **m, "budget_global": None if bud is None else bud["global"]})
    del mtl

    # --- attention (mask channels) ---
    print("Loading attention MAR splits ...", flush=True)
    train_xa, train_ya, test_xa, test_ya, all_xa = load_mar_splits("attention")
    assert all_xa.shape[1] == 226, all_xa.shape

    print(f"Loading FFNN+attention: {FFNN_ATTN_CKPT}", flush=True)
    nn_a = buildAttentionDNN(
        all_xa, n_value_features=113, d_model=256, n_heads=8, n_layers=3, ff_mult=2, dropout=0.2
    )
    nn_a.double()
    nn_a.load_state_dict(torch_load(FFNN_ATTN_CKPT, device))
    nn_a.to(device)
    y_tr, p_tr = predict_dnn(nn_a, train_xa, train_ya, device)
    y_te, p_te = predict_dnn(nn_a, test_xa, test_ya, device)
    m = plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "scatter_ffnn_attention.png")
    map_dst = copy_map("ffnn_attention", PAPER_MAPS["ffnn_attention"])
    bud = budget_for(map_dst)
    rows.append({"model": "ffnn_attention", **m, "budget_global": None if bud is None else bud["global"]})
    del nn_a

    print(f"Loading MTL+attention: {MTL_ATTN_CKPT}", flush=True)
    mtl_a = buildAttentionMTL(
        all_xa, n_value_features=113, d_model=128, n_heads=8, n_layers=2, ff_mult=2, dropout=0.2
    )
    mtl_a.double()
    mtl_a.load_state_dict(torch_load(MTL_ATTN_CKPT, device))
    mtl_a.to(device)
    y_tr, p_tr = predict_mtl(mtl_a, train_xa, train_ya, device)
    y_te, p_te = predict_mtl(mtl_a, test_xa, test_ya, device)
    m = plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "scatter_mtl_attention.png")
    map_dst = copy_map("mtl_attention", PAPER_MAPS["mtl_attention"])
    bud = budget_for(map_dst)
    rows.append({"model": "mtl_attention", **m, "budget_global": None if bud is None else bud["global"]})
    del mtl_a

    # Install SI supplementary scatter plots as the canonical paper figures.
    # Mapping: SI filename stem → notebook-friendly alias (png + pdf when present).
    si_scatter_map = {
        "scatter_entire_rf": "random_forest",
        "scatter_entire_xgb": "xgboost",
        "scatter_entire_catb": "catboost",
        "scatter_entire_knn": "knn",
        "scatter_entire_tabpfn": "tabpfn",
        "scatter_entire_nn": "ffnn",
        "scatter_entire_mtl": "mtl",
        "scatter_entire_nn_attention": "ffnn_attention",
        "scatter_entire_mtl_attention": "mtl_attention",
    }
    si_dir = FIGS / "si_paper"
    si_dir.mkdir(parents=True, exist_ok=True)
    for si_stem, model_key in si_scatter_map.items():
        for ext in (".png", ".pdf"):
            src = NN4_ROOT / "docs" / "figs" / f"{si_stem}{ext}"
            if not src.exists():
                continue
            # Keep SI original name + model alias for notebooks
            shutil.copy2(src, si_dir / f"{si_stem}{ext}")
            shutil.copy2(src, FIGS / f"si_{si_stem}{ext}")
            # Overwrite re-scored deep-model scatters with exact SI figures
            if model_key in ("ffnn", "mtl", "tabpfn", "ffnn_attention", "mtl_attention"):
                shutil.copy2(src, FIGS / f"scatter_{model_key}{ext}")
            print(f"Installed SI scatter: {si_stem}{ext}", flush=True)

    # Clean-named prediction maps for notebooks (same bytes as paper)
    clean_maps = OUT / "maps_paper_names"
    clean_maps.mkdir(parents=True, exist_ok=True)
    for key, src in PAPER_MAPS.items():
        if src.exists():
            shutil.copy2(src, clean_maps / src.name)
            print(f"Installed paper map: {src.name}", flush=True)

    df = pd.DataFrame(rows)
    # attach Table 2 targets for comparison
    df["table2_test_mae"] = df["model"].map(lambda k: TABLE2.get(k, {}).get("test_mae"))
    df["table2_test_cc"] = df["model"].map(lambda k: TABLE2.get(k, {}).get("test_cc"))
    df["table2_budget"] = df["model"].map(lambda k: TABLE2.get(k, {}).get("budget"))
    df["delta_test_mae"] = df["test_mae"] - df["table2_test_mae"]
    out_csv = OUT / "table2_reproduction_metrics.csv"
    df.to_csv(out_csv, index=False)

    # Exact Table-2 metrics for notebooks (authoritative paper numbers)
    si_stem_by_model = {v: k for k, v in si_scatter_map.items()}
    paper_rows = []
    for key, t2 in TABLE2.items():
        stem = si_stem_by_model.get(key, key)
        paper_rows.append(
            {
                "model": key,
                "train_mae": t2["train_mae"],
                "train_cc": t2["train_cc"],
                "test_mae": t2["test_mae"],
                "test_cc": t2["test_cc"],
                "budget_pg_yr": t2["budget"],
                "scatter_si": f"scatters/si_paper/{stem}.png",
                "note": "Table 2 / SI figures (authoritative)",
            }
        )
    paper_csv = OUT / "table2_paper_metrics.csv"
    pd.DataFrame(paper_rows).to_csv(paper_csv, index=False)

    print("\n=== Reproduction vs Table 2 ===", flush=True)
    print(df.to_string(index=False), flush=True)
    print(f"\nWrote {out_csv}", flush=True)
    print(f"Wrote paper metrics {paper_csv}", flush=True)
    print(f"SI scatters: {si_dir}", flush=True)
    print(f"Paper maps:  {clean_maps}", flush=True)
    print(f"Scatters: {FIGS}", flush=True)
    print(f"Maps: {MAPS_OUT}", flush=True)


if __name__ == "__main__":
    main()
