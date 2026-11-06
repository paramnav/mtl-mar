#!/usr/bin/env python3
"""
Step 3 — Train DNN (neural network) models, evaluate, produce scatter plots
and generate prediction maps with MC-dropout uncertainty.

Uses mtl-mar ``src/models/train_neural_network.py`` (MLP + attention).

Usage:
    python pipeline/step3_train_dnn.py
    python pipeline/step3_train_dnn.py --target MAR --arch attention
    python pipeline/step3_train_dnn.py --use-existing-features --skip-maps
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "src" / "models"))

import config as cfg
from step2_train_and_evaluate import (  # noqa: E402
    load_features,
    preprocess,
    save_scatter,
    set_seed,
    stratified_split,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_model(features, arch="mlp"):
    """Build a DNN using mtl-mar train_neural_network builders."""
    from train_neural_network import buildAttentionDNN, buildDNN

    if arch == "mlp":
        return buildDNN(cfg.DNN_LAYER_WIDTHS, features)
    if arch == "attention":
        return buildAttentionDNN(
            features,
            n_value_features=min(features.shape[1], cfg.N_ORIG_FEATURES),
            **cfg.ATTENTION_CFG,
        )
    raise ValueError(f"Unknown arch: {arch} (supported: mlp, attention)")


def train_dnn_model(train_features, train_labels, test_features, test_labels, arch="mlp"):
    """Train a DNN and return the trained model + predictions."""
    import train_neural_network as tnn
    from train_neural_network import create_dataloaders, train_model

    tnn.training_accuracy = []
    tnn.training_losses = []
    tnn.training_mae = []
    tnn.learning_rate = []
    tnn.evaluation_losses = []
    tnn.evaluation_mae = []
    tnn.true_labels = []
    tnn.pred_labels = []

    trainloader, testloader = create_dataloaders(
        train_features,
        train_labels,
        test_features,
        test_labels,
        batch_size=cfg.DNN_BATCH_SIZE,
    )

    model = build_model(train_features, arch=arch)
    if torch.cuda.is_available():
        model = model.cuda()

    # Paper attention / FFNN path uses Huber; keep Huber for both for consistency
    loss_fn = torch.nn.HuberLoss()
    train_model(
        model,
        loss_fn,
        trainloader,
        testloader,
        num_epochs=cfg.DNN_EPOCHS,
        lr=cfg.DNN_LR,
        weight_decay=cfg.DNN_WEIGHT_DECAY,
    )

    model.eval()

    def predict_numpy(model, X):
        X_t = torch.tensor(X, dtype=torch.float64).to(device)
        with torch.no_grad():
            preds = model(X_t).cpu().numpy().flatten()
        return preds

    tr_pred = predict_numpy(model, train_features)
    te_pred = predict_numpy(model, test_features)
    return model, tr_pred, te_pred


def predict_map_dnn(model, X_mean, X_std, tag, model_name, out_dir, is_mar=True, mc_samples=None):
    """Generate prediction map using the DNN (optional MC-dropout uncertainty)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(f for f in cfg.WORLD_FEATURES_DIR.iterdir() if f.suffix == ".npy")
    prediction_map = np.zeros((cfg.N_LAT, cfg.N_LON))

    if mc_samples and mc_samples > 1:
        uncertainty_map = np.zeros((cfg.N_LAT, cfg.N_LON))
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()
    else:
        uncertainty_map = None

    n_feat = X_mean.shape[0]
    n_orig = cfg.N_ORIG_FEATURES
    for chunk_idx, fpath in enumerate(files):
        chunk = np.load(fpath)
        if n_feat == 2 * n_orig and chunk.shape[1] >= 2 * n_orig:
            values = chunk[:, :n_orig]
            masks = chunk[:, n_orig : 2 * n_orig]
            values = np.nan_to_num(
                np.divide((values - X_mean[:n_orig]), X_std[:n_orig]),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            masks = np.nan_to_num(masks, nan=0.0, posinf=0.0, neginf=0.0)
            chunk_norm = np.concatenate([values, masks], axis=1)
        else:
            if chunk.shape[1] > n_feat:
                chunk = chunk[:, :n_feat]
            chunk_norm = np.nan_to_num(
                np.divide((chunk - X_mean), X_std),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        chunk_t = torch.tensor(chunk_norm, dtype=torch.float64).to(device)

        with torch.no_grad():
            if mc_samples and mc_samples > 1:
                preds_all = np.zeros((mc_samples, *cfg.CHUNK_SHAPE))
                for s in range(mc_samples):
                    preds_all[s] = model(chunk_t).cpu().numpy().reshape(cfg.CHUNK_SHAPE)
                preds = preds_all.mean(axis=0)
                var = preds_all.var(axis=0)
                row_s = chunk_idx * cfg.CHUNK_SHAPE[0]
                row_e = row_s + cfg.CHUNK_SHAPE[0]
                uncertainty_map[row_s:row_e, :] = var
            else:
                preds = model(chunk_t).cpu().numpy().reshape(cfg.CHUNK_SHAPE)

        row_s = chunk_idx * cfg.CHUNK_SHAPE[0]
        row_e = row_s + cfg.CHUNK_SHAPE[0]
        prediction_map[row_s:row_e, :] = preds

    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))

    land_map = np.load(cfg.LAND_MAP_FILE)
    land_map[np.isnan(land_map)] = 1
    land_mask = land_map.astype(bool)
    prediction_map[land_mask] = 0

    npy_path = out_dir / f"prediction_map_{tag}_{model_name}.npy"
    np.save(npy_path, prediction_map)

    fig, ax = plt.subplots(figsize=cfg.MAP_FIGSIZE, dpi=cfg.MAP_DPI)
    if is_mar:
        im = ax.imshow(
            prediction_map,
            norm=mcolors.LogNorm(vmin=cfg.MAP_VMIN_MAR, vmax=cfg.MAP_VMAX_MAR),
            cmap=cfg.MAP_CMAP_MAR,
            extent=[-180, 180, -90, 90],
        )
    else:
        pm = prediction_map.copy()
        pm[pm == 0] = np.nan
        im = ax.imshow(
            pm,
            vmin=cfg.MAP_VMIN_TOC,
            vmax=cfg.MAP_VMAX_TOC,
            cmap=cfg.MAP_CMAP_TOC,
            extent=[-180, 180, -90, 90],
        )
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    ax.set_title(f"{model_name} — {tag}")
    fig.tight_layout()
    fig.savefig(out_dir / f"prediction_map_{tag}_{model_name}.png")
    plt.close(fig)

    if uncertainty_map is not None:
        uncertainty_map = np.rot90(np.rot90(np.fliplr(uncertainty_map)))
        uncertainty_map[land_mask] = 0
        np.save(out_dir / f"uncertainty_map_{tag}_{model_name}.npy", uncertainty_map)
        fig, ax = plt.subplots(figsize=cfg.MAP_FIGSIZE, dpi=cfg.MAP_DPI)
        um = uncertainty_map.copy()
        um[um == 0] = np.nan
        im = ax.imshow(um, cmap="inferno", extent=[-180, 180, -90, 90])
        plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02, label="Prediction Variance")
        ax.set_title(f"Uncertainty (MC-Dropout) — {model_name} — {tag}")
        fig.tight_layout()
        fig.savefig(out_dir / f"uncertainty_map_{tag}_{model_name}.png")
        plt.close(fig)
        print("  → uncertainty map saved")

    print(f"  → DNN map saved: prediction_map_{tag}_{model_name}.png")
    gc.collect()


def run_dnn_for_target(target: str, arch: str, skip_maps: bool, use_existing: bool, mc_samples: int):
    is_mar = target.upper().startswith("MAR")
    cs_key, do_key = f"{target}_CS", f"{target}_DO"

    print(f"\n{'='*80}")
    print(f"  DNN Pipeline: {target} (arch={arch})")
    print(f"{'='*80}\n")

    # Attention prefers 226-d masked features when available
    values_only = arch != "attention"
    feat_CS, lab_CS, mean_CS, std_CS, lat_CS, lon_CS = load_features(
        cs_key, use_existing, values_only=values_only
    )
    feat_DO, lab_DO, mean_DO, std_DO, lat_DO, lon_DO = load_features(
        do_key, use_existing, values_only=values_only
    )
    X_mean, X_std = mean_DO, std_DO

    # Attention 226-d: z-score value channels; leave mask bits (std=1, mean=0)
    if not values_only and feat_CS.shape[1] == 2 * cfg.N_ORIG_FEATURES:
        n = cfg.N_ORIG_FEATURES
        mean_full = np.concatenate([X_mean[:n], np.zeros(n)])
        std_full = np.concatenate([X_std[:n], np.ones(n)])
        feat_CS, lab_CS, lat_CS, lon_CS = preprocess(
            feat_CS, lab_CS, lat_CS, lon_CS, mean_full, std_full
        )
        feat_DO, lab_DO, lat_DO, lon_DO = preprocess(
            feat_DO, lab_DO, lat_DO, lon_DO, mean_full, std_full
        )
        X_mean, X_std = mean_full, std_full
    else:
        feat_CS, lab_CS, lat_CS, lon_CS = preprocess(
            feat_CS, lab_CS, lat_CS, lon_CS, X_mean, X_std
        )
        feat_DO, lab_DO, lat_DO, lon_DO = preprocess(
            feat_DO, lab_DO, lat_DO, lon_DO, X_mean, X_std
        )

    trF_CS, trL_CS, _, _, teF_CS, teL_CS, _, _ = stratified_split(feat_CS, lab_CS, lat_CS, lon_CS)
    trF_DO, trL_DO, _, _, teF_DO, teL_DO, _, _ = stratified_split(feat_DO, lab_DO, lat_DO, lon_DO)

    train_features = np.concatenate([trF_CS, trF_DO])
    test_features = np.concatenate([teF_CS, teF_DO])
    train_labels = np.concatenate([trL_CS, trL_DO])
    test_labels = np.concatenate([teL_CS, teL_DO])

    print(
        f"Train: {train_features.shape[0]}, Test: {test_features.shape[0]}, "
        f"Features: {train_features.shape[1]}"
    )

    tag = f"{target}_entire"
    model_name = f"dnn_{arch}"

    set_seed()
    model, tr_pred, te_pred = train_dnn_model(
        train_features, train_labels, test_features, test_labels, arch=arch
    )

    tr_m, te_m = save_scatter(
        train_labels, tr_pred, test_labels, te_pred, model_name, tag, cfg.PIPELINE_SCATTER_DIR
    )

    results = {
        model_name: {
            "train": tr_m,
            "test": te_m,
            "arch": arch,
            "layers": cfg.DNN_LAYER_WIDTHS,
            "epochs": cfg.DNN_EPOCHS,
        }
    }

    cfg.PIPELINE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), cfg.PIPELINE_MODELS_DIR / f"model_{tag}_{model_name}.pth")
    print(f"  → model saved: model_{tag}_{model_name}.pth")

    if not skip_maps:
        predict_map_dnn(
            model,
            X_mean,
            X_std,
            tag,
            model_name,
            cfg.PIPELINE_MAPS_DIR,
            is_mar=is_mar,
            mc_samples=mc_samples,
        )

    cfg.PIPELINE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(cfg.PIPELINE_RESULTS_DIR / f"results_{tag}_{model_name}.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)


def main():
    parser = argparse.ArgumentParser(description="Step 3: Train DNN models")
    parser.add_argument("--target", type=str, default=None, choices=["MAR", "TOC"])
    parser.add_argument(
        "--arch",
        type=str,
        default="mlp",
        choices=["mlp", "attention"],
        help="mlp = paper FFNN [128]^3; attention = paper FFNN+attention",
    )
    parser.add_argument("--skip-maps", action="store_true")
    parser.add_argument("--use-existing-features", action="store_true")
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=cfg.MC_DROPOUT_SAMPLES,
        help="MC-dropout samples for uncertainty (0 to disable)",
    )
    args = parser.parse_args()

    targets = [args.target] if args.target else ["MAR", "TOC"]
    for t in targets:
        run_dnn_for_target(t, args.arch, args.skip_maps, args.use_existing_features, args.mc_samples)


if __name__ == "__main__":
    main()
