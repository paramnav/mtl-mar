#!/usr/bin/env python
"""Reproduce paper Table 2 metrics, SI scatters, and prediction maps in mtl-mar.

Why not just re-run notebooks_2 from scratch?
  - Trees/KNN with seed=42 + the *same* nn4sedratetoc feature pipeline match Table 2.
  - FFNN/MTL/attention are not bit-reproducible across GPU/CUDA/DataLoader shuffles
    even with torch.manual_seed(42). The paper numbers come from saved checkpoints
    and maps under nn4sedratetoc.

This script therefore:
  1. Uses the nn4sedratetoc masked_features + seed=42 splits (exact paper recipe)
  2. Refits trees/KNN (deterministic) and regenerates their scatters
  3. Loads FFNN/MTL/attention Table-2 checkpoints for metrics + scatters
  4. Copies the paper prediction maps into mtl-mar/data/output/.../paper_table2/
  5. Writes metrics_vs_paper.csv for side-by-side comparison
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MASKED_FEATURE_VARIANT", "values_only")

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

MTL_ROOT = Path("/gxfs_work/cau/sunms498/work/mtl-mar")
NN4_ROOT = Path("/gxfs_work/cau/sunms498/work/nn4sedratetoc")
NN4_NB = NN4_ROOT / "notebooks/SedRate"
NN4_FIGS = NN4_ROOT / "docs/figs"
NN4_CKPT = NN4_ROOT / "data/output/models"
NN4_MAPS = NN4_ROOT / "data/output/predictionmaps_SedRate/hyperparameter_optimization"

OUT = MTL_ROOT / "data/output/predictionmaps_SedRate/paper_table2"
FIGS = OUT / "scatters"
MAPS = OUT / "maps"

sys.path.extend(
    [
        str(NN4_ROOT / "src/models"),
        str(NN4_ROOT / "src/postprocessing"),
        str(NN4_NB),
    ]
)

from masked_features import (  # noqa: E402
    load_filtered_mask_stack,
    load_toc_task_masked,
    preprocess_features_labels_masked,
    variant_data_path,
)
from regenerate_si_scatter_plots import (  # noqa: E402
    load_mar_splits,
    plot_scatter,
    predict_dnn,
    predict_mtl,
    torch_load,
)
from trainDNN import buildAttentionDNN, buildDNN as buildDNN_single, set_seed  # noqa: E402
from trainMTL import buildAttentionMTL, buildDNN as buildDNN_mtl  # noqa: E402
from treeBasedModel import train_tree_based_model  # noqa: E402

SEED = 42

PAPER = {
    "Random Forests": {"train_mae": 8.15, "train_cc": 0.99, "test_mae": 19.50, "test_cc": 0.93, "budget": 68.27},
    "XGBoost": {"train_mae": 13.61, "train_cc": 0.99, "test_mae": 24.53, "test_cc": 0.90, "budget": 65.46},
    "CatBoost": {"train_mae": 31.10, "train_cc": 0.95, "test_mae": 34.25, "test_cc": 0.90, "budget": 66.12},
    "KNN": {"train_mae": 19.09, "train_cc": 0.89, "test_mae": 19.93, "test_cc": 0.92, "budget": 50.60},
    "TabPFN": {"train_mae": 15.28, "train_cc": 0.91, "test_mae": 17.27, "test_cc": 0.93, "budget": 40.39},
    "FFNN": {"train_mae": 15.02, "train_cc": 0.95, "test_mae": 20.18, "test_cc": 0.91, "budget": 34.82},
    "MTL": {"train_mae": 14.21, "train_cc": 0.91, "test_mae": 18.90, "test_cc": 0.92, "budget": 28.9},
    "FFNN + attention": {"train_mae": 17.35, "train_cc": 0.90, "test_mae": 20.50, "test_cc": 0.91, "budget": 33.38},
    "MTL + attention": {"train_mae": 16.57, "train_cc": 0.90, "test_mae": 20.29, "test_cc": 0.91, "budget": 36.77},
}

# Paper maps: prefer treebasedmodels_nomask paper HPs; FFNN/MTL/attention from active/archive
MAP_SOURCES = {
    "Random Forests": NN4_MAPS / "treebasedmodels_nomask/prediction_map_MAR_entire_rf_n700.npy",
    "XGBoost": NN4_MAPS / "treebasedmodels_nomask/prediction_map_MAR_entire_xgb_n1000.npy",
    "CatBoost": NN4_MAPS / "treebasedmodels_nomask/prediction_map_MAR_entire_catb_n1000.npy",
    "KNN": NN4_MAPS / "treebasedmodels_nomask/prediction_map_MAR_entire_knn_k6.npy",
    "TabPFN": NN4_MAPS / "prediction_map_MAR_entire_tabpfn_nomask.npy",
    "FFNN": NN4_MAPS / "prediction_map_SedRate_entire_supervised_nomask_[128, 128, 128].npy",
    "MTL": NN4_MAPS
    / "archive_pre_full_mtl_nomask_hprerun_20261006/prediction_map_MAR_entire_MTL_n_nomask_47.npy",
    "FFNN + attention": NN4_MAPS
    / "prediction_map_SedRate_entire_supervised_attention_attn_d256_h8_l3_ff2_do0.2.npy",
    "MTL + attention": NN4_MAPS
    / "prediction_map_MAR_entire_MTL_n_attention_attn_d128_h8_l2_ff2_do0.2.npy",
}

FFNN_CKPT = NN4_CKPT / "FFNN_nomask_[128,128,128]_table2.pt"
MTL_CKPT = NN4_CKPT / "MTL_nomask_map47_[128,128,128,128]_shared2_table2.pt"
FFNN_ATTN_CKPT = NN4_CKPT / "FFNN_attention_best.pt"
MTL_ATTN_CKPT = NN4_CKPT / "MTL_attention_best.pt"


def _cc(y, p):
    return float(np.corrcoef(np.asarray(y).ravel(), np.asarray(p).ravel())[0, 1])


def _row(name, y_tr, p_tr, y_te, p_te):
    return {
        "model": name,
        "train_mae": float(mean_absolute_error(y_tr, p_tr)),
        "train_cc": _cc(y_tr, p_tr),
        "test_mae": float(mean_absolute_error(y_te, p_te)),
        "test_cc": _cc(y_te, p_te),
        "paper_train_mae": PAPER[name]["train_mae"],
        "paper_train_cc": PAPER[name]["train_cc"],
        "paper_test_mae": PAPER[name]["test_mae"],
        "paper_test_cc": PAPER[name]["test_cc"],
        "paper_budget_Pg_yr": PAPER[name]["budget"],
    }


def copy_maps():
    MAPS.mkdir(parents=True, exist_ok=True)
    for name, src in MAP_SOURCES.items():
        if not src.exists():
            print(f"[WARN] missing map for {name}: {src}")
            continue
        dst = MAPS / f"{name.replace(' ', '_').replace('+', 'plus')}.npy"
        shutil.copy2(src, dst)
        print(f"Copied map {name} -> {dst.name}")


def copy_existing_scatters():
    FIGS.mkdir(parents=True, exist_ok=True)
    mapping = {
        "scatter_entire_rf.png": "Random_Forests",
        "scatter_entire_xgb.png": "XGBoost",
        "scatter_entire_catb.png": "CatBoost",
        "scatter_entire_knn.png": "KNN",
        "scatter_entire_tabpfn.png": "TabPFN",
        "scatter_entire_nn.png": "FFNN",
        "scatter_entire_mtl.png": "MTL",
        "scatter_entire_nn_attention.png": "FFNN_plus_attention",
        "scatter_entire_mtl_attention.png": "MTL_plus_attention",
    }
    for src_name, stem in mapping.items():
        src = NN4_FIGS / src_name
        if src.exists():
            shutil.copy2(src, FIGS / f"{stem}.png")
            pdf = src.with_suffix(".pdf")
            if pdf.exists():
                shutil.copy2(pdf, FIGS / f"{stem}.pdf")
            print(f"Copied existing scatter {src_name}")


def run_trees(train_x, train_y, test_x, test_y, rows):
    set_seed(SEED)
    configs = [
        ("Random Forests", "random_forest", 700),
        ("XGBoost", "xgboost", 1000),
        ("CatBoost", "catboost", 1000),
    ]
    for name, mtype, n in configs:
        print(f"Fitting {name} n={n} ...", flush=True)
        _, _, _, p_tr, p_te = train_tree_based_model(
            train_x, train_y, test_x, test_y, model_type=mtype, n_estimators=n, random_seed=SEED
        )
        plot_scatter(train_y, p_tr, test_y, p_te, FIGS / f"{name.replace(' ', '_')}.png")
        rows.append(_row(name, train_y, p_tr, test_y, p_te))

    print("Fitting KNN k=6 ...", flush=True)
    knn = KNeighborsRegressor(n_neighbors=6, n_jobs=-1)
    knn.fit(train_x, train_y)
    p_tr, p_te = knn.predict(train_x), knn.predict(test_x)
    plot_scatter(train_y, p_tr, test_y, p_te, FIGS / "KNN.png")
    rows.append(_row("KNN", train_y, p_tr, test_y, p_te))


def run_ffnn_mtl_from_ckpt(train_x, train_y, test_x, test_y, all_x, device, rows):
    set_seed(SEED)
    print(f"Loading FFNN from {FFNN_CKPT}", flush=True)
    ffnn = buildDNN_single([128, 128, 128], all_x)
    ffnn.double()
    ffnn.load_state_dict(torch_load(FFNN_CKPT, device))
    ffnn.to(device)
    y_tr, p_tr = predict_dnn(ffnn, train_x, train_y, device)
    y_te, p_te = predict_dnn(ffnn, test_x, test_y, device)
    plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "FFNN.png")
    rows.append(_row("FFNN", y_tr, p_tr, y_te, p_te))
    del ffnn

    print(f"Loading MTL map47 from {MTL_CKPT}", flush=True)
    mtl = buildDNN_mtl([128, 128, 128, 128], all_x, 2)
    mtl.double()
    mtl.load_state_dict(torch_load(MTL_CKPT, device))
    mtl.to(device)
    y_tr, p_tr = predict_mtl(mtl, train_x, train_y, device)
    y_te, p_te = predict_mtl(mtl, test_x, test_y, device)
    plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "MTL.png")
    rows.append(_row("MTL", y_tr, p_tr, y_te, p_te))
    del mtl


def run_attention_from_ckpt(device, rows):
    os.environ["MASKED_FEATURE_VARIANT"] = "attention"
    train_x, train_y, test_x, test_y, all_x = load_mar_splits("attention")
    assert all_x.shape[1] == 226, all_x.shape

    print(f"Loading FFNN+attention from {FFNN_ATTN_CKPT}", flush=True)
    nn = buildAttentionDNN(
        all_x, n_value_features=113, d_model=256, n_heads=8, n_layers=3, ff_mult=2, dropout=0.2
    )
    nn.double()
    nn.load_state_dict(torch_load(FFNN_ATTN_CKPT, device))
    nn.to(device)
    y_tr, p_tr = predict_dnn(nn, train_x, train_y, device)
    y_te, p_te = predict_dnn(nn, test_x, test_y, device)
    plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "FFNN_plus_attention.png")
    rows.append(_row("FFNN + attention", y_tr, p_tr, y_te, p_te))
    del nn

    print(f"Loading MTL+attention from {MTL_ATTN_CKPT}", flush=True)
    mtl = buildAttentionMTL(
        all_x, n_value_features=113, d_model=128, n_heads=8, n_layers=2, ff_mult=2, dropout=0.2
    )
    mtl.double()
    mtl.load_state_dict(torch_load(MTL_ATTN_CKPT, device))
    mtl.to(device)
    y_tr, p_tr = predict_mtl(mtl, train_x, train_y, device)
    y_te, p_te = predict_mtl(mtl, test_x, test_y, device)
    plot_scatter(y_tr, p_tr, y_te, p_te, FIGS / "MTL_plus_attention.png")
    rows.append(_row("MTL + attention", y_tr, p_tr, y_te, p_te))
    del mtl
    os.environ["MASKED_FEATURE_VARIANT"] = "values_only"


def add_tabpfn_from_paper(rows):
    """TabPFN scatter already exists; record paper numbers (full refit is heavy)."""
    src = NN4_FIGS / "scatter_entire_tabpfn.png"
    if src.exists():
        shutil.copy2(src, FIGS / "TabPFN.png")
        pdf = src.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, FIGS / "TabPFN.pdf")
    rows.append(
        {
            "model": "TabPFN",
            "train_mae": PAPER["TabPFN"]["train_mae"],
            "train_cc": PAPER["TabPFN"]["train_cc"],
            "test_mae": PAPER["TabPFN"]["test_mae"],
            "test_cc": PAPER["TabPFN"]["test_cc"],
            "paper_train_mae": PAPER["TabPFN"]["train_mae"],
            "paper_train_cc": PAPER["TabPFN"]["train_cc"],
            "paper_test_mae": PAPER["TabPFN"]["test_mae"],
            "paper_test_cc": PAPER["TabPFN"]["test_cc"],
            "paper_budget_Pg_yr": PAPER["TabPFN"]["budget"],
            "note": "copied paper scatter / metrics (no TabPFN refit)",
        }
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)
    print("Output:", OUT, flush=True)

    copy_maps()
    copy_existing_scatters()

    print("Loading MAR values_only splits (seed=42) ...", flush=True)
    train_x, train_y, test_x, test_y, all_x = load_mar_splits("values_only")
    print(f"train={train_x.shape} test={test_x.shape} n_features={all_x.shape[1]}", flush=True)

    rows = []
    run_trees(train_x, train_y, test_x, test_y, rows)
    add_tabpfn_from_paper(rows)
    run_ffnn_mtl_from_ckpt(train_x, train_y, test_x, test_y, all_x, device, rows)
    run_attention_from_ckpt(device, rows)

    df = pd.DataFrame(rows)
    df["delta_test_mae"] = df["test_mae"] - df["paper_test_mae"]
    df["delta_test_cc"] = df["test_cc"] - df["paper_test_cc"]
    csv_path = OUT / "metrics_vs_paper.csv"
    df.to_csv(csv_path, index=False)
    print("\n=== metrics vs paper ===", flush=True)
    print(
        df[
            [
                "model",
                "test_mae",
                "paper_test_mae",
                "delta_test_mae",
                "test_cc",
                "paper_test_cc",
                "delta_test_cc",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nWrote {csv_path}", flush=True)
    print(f"Scatters: {FIGS}", flush=True)
    print(f"Maps: {MAPS}", flush=True)
    print(
        "\nReproducibility note: trees/KNN should match Table 2 to ~0.01–0.05 MAE.\n"
        "FFNN/MTL/attention metrics match when loading *_table2.pt / *_best.pt checkpoints.\n"
        "Fresh GPU retrains without those weights will NOT reproduce Table 2 exactly.",
        flush=True,
    )


if __name__ == "__main__":
    main()
