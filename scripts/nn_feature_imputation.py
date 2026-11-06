"""Neural-network cross-feature imputation for ocean gap cells (paper path).

Shared cube / native-mask helpers live in ``ml_feature_imputation.py``.
Data contract:

  * Models are trained only on ocean cells where the target feature is
    natively observed.
  * Predictors are all other stack features, masked to NaN wherever that
    predictor is itself a gap cell — imputed values are NEVER used as
    predictors.
  * Imputed targets are never chained into later features (each feature is
    imputed independently from the original native-only cube).

Because MLPs cannot ingest NaN, each masked predictor is represented as
(z-scored value with NaN -> 0) plus an explicit binary missing-indicator
channel, so the network can distinguish "missing" from "average value".
The z-score statistics are computed on the training split only.

Hyperparameters are selected per target feature by random search, scored by
R^2 on a held-out validation split of native cells (never seen during
training).  The best configuration is then retrained on the full training
split and final validation metrics (R^2 / RMSE / MAE) are reported in the
per-feature stats, which harmonize_ocean_gaps.py writes to
ml_imputation_summary.csv.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from ml_feature_imputation import (
    DEFAULT_MAX_TRAIN_SAMPLES,
    DEFAULT_MIN_NATIVE_PREDICTORS,
    _build_geo_grids,
    _clip_rule_for,
    _clip_values,
    _is_circular_direction,
    _predictor_indices,
    base_features_with_gaps,
    count_native_predictors,
    load_stack_cube,
    mask_predictors_to_native,
    native_mask_for_feature,
)
from spatial_interpolation import (
    DEFAULT_POLAR_LAT_THRESHOLD_DEG,
    fill_high_latitude_gaps_idw,
    fill_ocean_gaps,
    high_latitude_ocean_mask,
    smooth_circular_ocean_gaps,
)

DEFAULT_HP_TRIALS = 8
DEFAULT_VAL_SIZE = 100_000
SEARCH_MAX_SAMPLES = 150_000  # subsample per search trial (speed)
SEARCH_MAX_EPOCHS = 60
FINAL_MAX_EPOCHS = 200
SEARCH_PATIENCE = 8
FINAL_PATIENCE = 15
PREDICT_BATCH = 65_536

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEARCH_SPACE: dict[str, list] = {
    "hidden": [(256, 128, 64), (512, 256, 128), (128, 64), (512, 256, 128, 64), (256, 256, 128)],
    "dropout": [0.0, 0.1, 0.2],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
    "batch_size": [4096, 8192, 16384],
}


@dataclass(frozen=True)
class NNConfig:
    hidden: tuple[int, ...]
    dropout: float
    lr: float
    weight_decay: float
    batch_size: int


class MLPImputer(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------

def fit_standardizer(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise mean/std on finite training values only."""
    mu = np.nanmean(x_train, axis=0)
    sd = np.nanstd(x_train, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def prepare_inputs(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """[z-scored values (NaN->0) | binary missing indicators] as float32."""
    missing = ~np.isfinite(x)
    z = (x - mu) / sd
    z[missing] = 0.0
    return np.concatenate([z.astype(np.float32), missing.astype(np.float32)], axis=1)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    resid = y_pred - y_true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "r2": r2,
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _predict_batched(model: MLPImputer, x: torch.Tensor) -> np.ndarray:
    model.eval()
    out = np.empty(x.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, x.shape[0], PREDICT_BATCH):
            out[i : i + PREDICT_BATCH] = model(x[i : i + PREDICT_BATCH].to(DEVICE)).cpu().numpy()
    return out


def train_mlp(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: NNConfig,
    *,
    max_epochs: int,
    patience: int,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[MLPImputer, dict[str, float], float, float]:
    """Train one MLP with early stopping on validation MSE.

    The target is z-scored internally (helps optimization for features with
    large scales); returned metrics and predictions are in original units.
    Returns (model, val_metrics, y_mu, y_sd).
    """
    torch.manual_seed(seed)
    y_mu = float(y_fit.mean())
    y_sd = float(y_fit.std())
    if not math.isfinite(y_sd) or y_sd < 1e-9:
        y_sd = 1.0

    xt = torch.tensor(x_fit)
    yt = torch.tensor(((y_fit - y_mu) / y_sd).astype(np.float32))
    xv = torch.tensor(x_val)

    model = MLPImputer(x_fit.shape[1], cfg.hidden, cfg.dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    lossf = nn.MSELoss()

    n = xt.shape[0]
    best_val = float("inf")
    best_state: dict | None = None
    bad = 0
    gen = torch.Generator().manual_seed(seed)

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            if idx.numel() < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = lossf(model(xt[idx].to(DEVICE)), yt[idx].to(DEVICE))
            loss.backward()
            opt.step()

        pred_val = _predict_batched(model, xv) * y_sd + y_mu
        val_mse = float(np.mean((pred_val - y_val) ** 2))
        if val_mse < best_val - 1e-9:
            best_val = val_mse
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if progress and ((epoch + 1) % 10 == 0 or epoch == 0):
            progress(f"        epoch {epoch + 1}/{max_epochs}  val_mse={val_mse:.6g}  best={best_val:.6g}")
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    pred_val = _predict_batched(model, xv) * y_sd + y_mu
    return model, _metrics(y_val, pred_val), y_mu, y_sd


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------

def sample_configs(n_trials: int, rng: np.random.Generator) -> list[NNConfig]:
    """Random search over SEARCH_SPACE (unique configs, baseline always included)."""
    baseline = NNConfig(hidden=(256, 128, 64), dropout=0.1, lr=1e-3, weight_decay=1e-5, batch_size=8192)
    configs: list[NNConfig] = [baseline]
    seen = {baseline}
    attempts = 0
    while len(configs) < n_trials and attempts < 200:
        attempts += 1
        cfg = NNConfig(
            hidden=tuple(SEARCH_SPACE["hidden"][rng.integers(len(SEARCH_SPACE["hidden"]))]),
            dropout=float(SEARCH_SPACE["dropout"][rng.integers(len(SEARCH_SPACE["dropout"]))]),
            lr=float(SEARCH_SPACE["lr"][rng.integers(len(SEARCH_SPACE["lr"]))]),
            weight_decay=float(SEARCH_SPACE["weight_decay"][rng.integers(len(SEARCH_SPACE["weight_decay"]))]),
            batch_size=int(SEARCH_SPACE["batch_size"][rng.integers(len(SEARCH_SPACE["batch_size"]))]),
        )
        if cfg not in seen:
            seen.add(cfg)
            configs.append(cfg)
    return configs[:n_trials]


def search_hyperparameters(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_trials: int,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[NNConfig, list[dict]]:
    """Random search scored by held-out validation R^2. Returns (best_cfg, trials)."""
    rng = np.random.default_rng(seed)
    configs = sample_configs(n_trials, rng)

    # Subsample the fit split for speed during the search phase only.
    if x_fit.shape[0] > SEARCH_MAX_SAMPLES:
        sub = rng.choice(x_fit.shape[0], size=SEARCH_MAX_SAMPLES, replace=False)
        x_sub, y_sub = x_fit[sub], y_fit[sub]
    else:
        x_sub, y_sub = x_fit, y_fit

    trials: list[dict] = []
    best_cfg = configs[0]
    best_r2 = -float("inf")
    for k, cfg in enumerate(configs):
        if progress:
            progress(
                f"    trial {k + 1}/{len(configs)}: hidden={cfg.hidden} dropout={cfg.dropout} "
                f"lr={cfg.lr} wd={cfg.weight_decay} batch={cfg.batch_size}"
            )
        _, m, _, _ = train_mlp(
            x_sub, y_sub, x_val, y_val, cfg,
            max_epochs=SEARCH_MAX_EPOCHS, patience=SEARCH_PATIENCE, seed=seed + k,
        )
        trials.append({**asdict(cfg), **{f"val_{key}": v for key, v in m.items()}})
        if progress:
            progress(f"      val R2={m['r2']:.4f}  RMSE={m['rmse']:.4g}  MAE={m['mae']:.4g}")
        if m["r2"] > best_r2:
            best_r2 = m["r2"]
            best_cfg = cfg
    return best_cfg, trials


# ---------------------------------------------------------------------------
# Per-feature imputation
# ---------------------------------------------------------------------------

def _split_train_val(
    n: int,
    *,
    max_train_samples: int,
    val_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint (fit_idx, val_idx) into a pool of n native cells."""
    perm = rng.permutation(n)
    n_val = min(val_size, n // 5)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]
    if fit_idx.size > max_train_samples:
        fit_idx = fit_idx[:max_train_samples]
    return fit_idx, val_idx


def impute_feature_nn(
    target_idx: int,
    names: list[str],
    cube: np.ndarray,
    gap_stack: np.ndarray,
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
    min_native_predictors: int = DEFAULT_MIN_NATIVE_PREDICTORS,
    hp_trials: int = DEFAULT_HP_TRIALS,
    val_size: int = DEFAULT_VAL_SIZE,
    random_state: int = 42,
    polar_lat_threshold: float = DEFAULT_POLAR_LAT_THRESHOLD_DEG,
    progress: Callable[[str], None] | None = print,
) -> tuple[np.ndarray, dict]:
    """Impute gap cells for one target feature with a tuned MLP.

    Returns (filled_grid, stats) — stats includes validation metrics and the
    selected hyperparameters.
    """
    fname = names[target_idx]
    grid = cube[:, :, target_idx].copy()
    gap = gap_stack[:, :, target_idx].astype(bool)
    native = native_mask_for_feature(grid, gap, ocean)
    # Full gap mask (includes already-filled imputed cells, e.g. Arctic CHL POC)
    # stays for native/seed exclusion; only still-missing cells are written.
    flagged = gap & ocean
    needs_fill = flagged & ~np.isfinite(grid)
    high_lat = high_latitude_ocean_mask(lats, ocean.shape, threshold_deg=polar_lat_threshold)
    polar_gap = needs_fill & high_lat
    low_lat_gap = needs_fill & ~high_lat
    stats: dict = {
        "feature": fname,
        "gap_cells": int(flagged.sum()),
        "native_train_cells": int(native.sum()),
        "ml_filled": 0,
        "spatial_fallback": 0,
        "polar_idw_filled": 0,
        "already_filled_gap_cells": int((flagged & np.isfinite(grid)).sum()),
        "needs_fill_cells": int(needs_fill.sum()),
    }
    if not needs_fill.any():
        stats["method"] = "noop_already_filled"
        return grid, stats
    if native.sum() < 1000:
        filled = fill_ocean_gaps(grid, needs_fill, ocean, method="nearest")
        stats["spatial_fallback"] = int(needs_fill.sum())
        stats["method"] = "spatial_only_insufficient_native"
        return filled.astype(np.float32), stats

    pred_indices = _predictor_indices(target_idx, names)
    lat_grid, lon_grid = _build_geo_grids(lats, lons, ocean.shape)
    x_all, ocean_idx = mask_predictors_to_native(
        cube, gap_stack, ocean, pred_indices, lat_grid, lon_grid
    )

    flat_grid = grid.ravel()
    train_mask = native.ravel()[ocean_idx]
    gap_mask = needs_fill.ravel()[ocean_idx]

    x_pool = x_all[train_mask]
    y_pool = flat_grid[ocean_idx][train_mask].astype(np.float64)
    x_gap_raw = x_all[gap_mask]

    n_native_pred = count_native_predictors(x_gap_raw)
    gap_high_lat = high_lat.ravel()[ocean_idx][gap_mask]
    sparse = (n_native_pred < min_native_predictors) & ~gap_high_lat
    dense = (n_native_pred >= min_native_predictors) & ~gap_high_lat

    rng = np.random.default_rng(random_state)
    fit_idx, val_idx = _split_train_val(
        y_pool.shape[0], max_train_samples=max_train_samples, val_size=val_size, rng=rng
    )

    # Standardizer fitted on the training split only; NaN -> 0 + indicators.
    mu, sd = fit_standardizer(x_pool[fit_idx])
    x_fit = prepare_inputs(x_pool[fit_idx], mu, sd)
    x_val = prepare_inputs(x_pool[val_idx], mu, sd)

    clip_rule = _clip_rule_for(fname)
    out = grid.copy()

    if _is_circular_direction(fname):
        rad = np.deg2rad(y_pool)
        y_sin, y_cos = np.sin(rad), np.cos(rad)
        if progress:
            progress(f"    HP search on sin component ({hp_trials} trials, fit={fit_idx.size:,}, val={val_idx.size:,})")
        best_cfg, trials = search_hyperparameters(
            x_fit, y_sin[fit_idx], x_val, y_sin[val_idx],
            n_trials=hp_trials, seed=random_state, progress=progress,
        )
        if progress:
            progress(f"    final sin/cos training with {best_cfg}")
        model_sin, m_sin, mu_sin, sd_sin = train_mlp(
            x_fit, y_sin[fit_idx], x_val, y_sin[val_idx], best_cfg,
            max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, seed=random_state, progress=progress,
        )
        model_cos, m_cos, mu_cos, sd_cos = train_mlp(
            x_fit, y_cos[fit_idx], x_val, y_cos[val_idx], best_cfg,
            max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, seed=random_state + 1, progress=progress,
        )
        # Combined angular error on the validation split
        xv_t = torch.tensor(x_val)
        pv_sin = _predict_batched(model_sin, xv_t) * sd_sin + mu_sin
        pv_cos = _predict_batched(model_cos, xv_t) * sd_cos + mu_cos
        pred_deg = np.rad2deg(np.arctan2(pv_sin, pv_cos))
        true_deg = y_pool[val_idx]
        diff = np.abs((pred_deg - true_deg + 180.0) % 360.0 - 180.0)
        stats.update(
            val_r2=float((m_sin["r2"] + m_cos["r2"]) / 2.0),
            val_rmse=float(np.sqrt(np.mean(diff ** 2))),
            val_mae=float(np.mean(diff)),
            val_r2_sin=m_sin["r2"],
            val_r2_cos=m_cos["r2"],
        )
        if dense.any():
            xg = torch.tensor(prepare_inputs(x_gap_raw[dense], mu, sd))
            pred_sin = _predict_batched(model_sin, xg) * sd_sin + mu_sin
            pred_cos = _predict_batched(model_cos, xg) * sd_cos + mu_cos
            pred_deg = np.rad2deg(np.arctan2(pred_sin, pred_cos))
            pred_deg = (np.mod(pred_deg + 180.0, 360.0) - 180.0).astype(np.float32)
            gap_dense_idx = np.flatnonzero(gap_mask)[dense]
            out.ravel()[ocean_idx[gap_dense_idx]] = pred_deg
            stats["ml_filled"] = int(dense.sum())
    else:
        if progress:
            progress(f"    HP search ({hp_trials} trials, fit={fit_idx.size:,}, val={val_idx.size:,}, {x_fit.shape[1]} inputs)")
        best_cfg, trials = search_hyperparameters(
            x_fit, y_pool[fit_idx], x_val, y_pool[val_idx],
            n_trials=hp_trials, seed=random_state, progress=progress,
        )
        if progress:
            progress(f"    final training with {best_cfg}")
        model, m, y_mu, y_sd = train_mlp(
            x_fit, y_pool[fit_idx], x_val, y_pool[val_idx], best_cfg,
            max_epochs=FINAL_MAX_EPOCHS, patience=FINAL_PATIENCE, seed=random_state, progress=progress,
        )
        stats.update(val_r2=m["r2"], val_rmse=m["rmse"], val_mae=m["mae"])
        if dense.any():
            xg = torch.tensor(prepare_inputs(x_gap_raw[dense], mu, sd))
            pred = _predict_batched(model, xg) * y_sd + y_mu
            pred = _clip_values(pred, clip_rule).astype(np.float32)
            gap_dense_idx = np.flatnonzero(gap_mask)[dense]
            out.ravel()[ocean_idx[gap_dense_idx]] = pred
            stats["ml_filled"] = int(dense.sum())

    stats["best_config"] = str(asdict(best_cfg))
    stats["hp_trials"] = len(trials)

    if sparse.any():
        if progress:
            progress(f"    spatial fallback for {int(sparse.sum()):,} cells (<{min_native_predictors} native predictors)")
        partial = fill_ocean_gaps(
            out, needs_fill, ocean, method="nearest", circular=_is_circular_direction(fname)
        )
        gap_sparse_idx = np.flatnonzero(gap_mask)[sparse]
        out.ravel()[ocean_idx[gap_sparse_idx]] = partial.ravel()[ocean_idx[gap_sparse_idx]]
        stats["spatial_fallback"] = int(sparse.sum())

    if _is_circular_direction(fname) and low_lat_gap.any():
        if progress:
            progress(f"    circular boundary smoothing on {int(low_lat_gap.sum()):,} low-latitude gap cells")
        out = smooth_circular_ocean_gaps(out, needs_fill & ~high_lat, ocean)

    if polar_gap.any():
        if progress:
            progress(
                f"    polar IDW fill for {int(polar_gap.sum()):,} cells "
                f"(|lat|>={polar_lat_threshold:.0f}°)"
            )
        out, n_polar = fill_high_latitude_gaps_idw(
            out,
            gap,
            ocean,
            lats,
            threshold_deg=polar_lat_threshold,
            circular=_is_circular_direction(fname),
            write_mask=needs_fill,
        )
        stats["polar_idw_filled"] = n_polar

    out[native] = grid[native]
    # Preserve already-filled imputed cells (e.g. Arctic CHL) and land.
    keep = flagged & np.isfinite(grid) & ~needs_fill
    out[keep] = grid[keep]
    out[~ocean] = grid[~ocean]
    if stats["polar_idw_filled"]:
        base = "nn+polar_idw" if stats["ml_filled"] else "polar_idw"
        stats["method"] = f"{base}+spatial_fallback" if stats["spatial_fallback"] else base
    else:
        stats["method"] = "nn+spatial_fallback" if stats["spatial_fallback"] else "nn"
    return out.astype(np.float32), stats


def impute_stack_nn(
    names: list[str],
    gap_by_name: dict[str, np.ndarray],
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    only: list[str] | None = None,
    max_train_samples: int = DEFAULT_MAX_TRAIN_SAMPLES,
    min_native_predictors: int = DEFAULT_MIN_NATIVE_PREDICTORS,
    hp_trials: int = DEFAULT_HP_TRIALS,
    polar_lat_threshold: float = DEFAULT_POLAR_LAT_THRESHOLD_DEG,
    progress: Callable[[str], None] | None = print,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    """NN-impute all qualifying base features; returns {fname: grid} and stats rows."""
    targets = base_features_with_gaps(names, gap_by_name, ocean)
    if only:
        only_set = set(only)
        targets = [f for f in targets if f in only_set]
    if not targets:
        return {}, []

    if progress:
        progress(f"Loading {len(names)}-feature stack cube...")
        progress(f"Device: {DEVICE}")
    cube = load_stack_cube(names, ocean, lats, lons)
    gap_stack = np.stack([gap_by_name[f].astype(np.uint8) for f in names], axis=-1)

    # Sort smallest gaps first (independent models; order does not affect predictors).
    targets.sort(key=lambda f: int((gap_by_name[f] & ocean).sum()))

    filled: dict[str, np.ndarray] = {}
    rows: list[dict] = []
    for fname in targets:
        tidx = names.index(fname)
        if progress:
            progress(f"  NN impute: {fname}")
        grid, stats = impute_feature_nn(
            tidx,
            names,
            cube,
            gap_stack,
            ocean,
            lats,
            lons,
            max_train_samples=max_train_samples,
            min_native_predictors=min_native_predictors,
            hp_trials=hp_trials,
            polar_lat_threshold=polar_lat_threshold,
            progress=progress,
        )
        filled[fname] = grid
        rows.append(stats)
    return filled, rows
