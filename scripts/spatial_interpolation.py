"""Ocean-gap interpolation for gridded feature layers (2160 x 4320, south-up)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

WAVE_BASE_FEATURES = frozenset(
    {
        "SS_WAVE_HEIGHT_M_2018_2023_WAVEWATCH3x.5m.grd",
        "SS_WAVE_PERIOD_S_2018_2023_WAVEWATCH3x.5m.grd",
        "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd",
    }
)

# |latitude| >= threshold: gap cells are filled with Laplacian IDW from native
# ocean observations instead of cross-feature ML/NN (reduces polar banding).
DEFAULT_POLAR_LAT_THRESHOLD_DEG = 60.0


def is_wave_base_feature(fname: str) -> bool:
    return fname in WAVE_BASE_FEATURES


def is_wave_feature(fname: str) -> bool:
    return fname.startswith("SS_WAVE_")


def _nearest_from_valid(grid: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill non-valid cells with the nearest valid cell value."""
    invalid = ~valid
    if not invalid.any():
        return grid.astype(np.float64, copy=True)
    _, indices = distance_transform_edt(invalid, return_indices=True)
    return grid[tuple(indices)].astype(np.float64)


def _laplacian_refine(
    grid: np.ndarray,
    fill_mask: np.ndarray,
    valid: np.ndarray,
    *,
    n_iter: int = 250,
) -> np.ndarray:
    """Diffuse values into *fill_mask* from bordering valid cells (smooth IDW-like fill)."""
    out = grid.astype(np.float64, copy=True)
    pending = fill_mask.copy()
    for _ in range(n_iter):
        if not pending.any():
            break
        nbr = (
            np.roll(out, 1, axis=0)
            + np.roll(out, -1, axis=0)
            + np.roll(out, 1, axis=1)
            + np.roll(out, -1, axis=1)
        ) / 4.0
        nbr_valid = (
            np.roll(valid.astype(np.float64), 1, axis=0)
            + np.roll(valid.astype(np.float64), -1, axis=0)
            + np.roll(valid.astype(np.float64), 1, axis=1)
            + np.roll(valid.astype(np.float64), -1, axis=1)
        )
        can_fill = pending & (nbr_valid > 0)
        if not can_fill.any():
            break
        out[can_fill] = nbr[can_fill]
        valid = valid | can_fill
        pending = pending & ~can_fill
    return out


def _circular_nearest_fill(
    grid: np.ndarray,
    fill_mask: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor fill on sin/cos components (degrees in [-180, 180])."""
    rad = np.deg2rad(np.where(valid, grid, 0.0))
    sin_g = np.where(valid, np.sin(rad), 0.0)
    cos_g = np.where(valid, np.cos(rad), 0.0)
    sin_f = _nearest_from_valid(sin_g, valid)
    cos_f = _nearest_from_valid(cos_g, valid)
    out = np.rad2deg(np.arctan2(sin_f, cos_f))
    out = (np.mod(out + 180.0, 360.0) - 180.0).astype(np.float64)
    out[valid] = grid[valid]
    out[fill_mask] = ((np.mod(out[fill_mask] + 180.0, 360.0) - 180.0))
    return out


def _neighbor_mean_4(grid: np.ndarray, active: np.ndarray) -> np.ndarray:
    """4-neighbor mean using only *active* cells (e.g. ocean); inactive neighbors ignored."""
    stacks = (
        np.roll(grid, 1, axis=0),
        np.roll(grid, -1, axis=0),
        np.roll(grid, 1, axis=1),
        np.roll(grid, -1, axis=1),
    )
    masks = (
        np.roll(active, 1, axis=0),
        np.roll(active, -1, axis=0),
        np.roll(active, 1, axis=1),
        np.roll(active, -1, axis=1),
    )
    total = np.zeros_like(grid, dtype=np.float64)
    count = np.zeros_like(grid, dtype=np.float64)
    for val, mask in zip(stacks, masks):
        use = mask
        total += np.where(use, val, 0.0)
        count += use.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, grid)


def smooth_circular_ocean_gaps(
    grid: np.ndarray,
    gap: np.ndarray,
    ocean: np.ndarray,
    *,
    n_iter: int = 48,
    blend: float = 0.45,
) -> np.ndarray:
    """Relax imputed circular angles toward a local ocean-neighbor mean.

    Native (non-gap) ocean cells stay fixed; only ``gap & ocean`` cells are
    updated.  Diffusion runs on sin/cos components so ±180° seams are avoided.
    """
    fill_mask = gap & ocean
    native = (~gap) & ocean & np.isfinite(grid)
    if not fill_mask.any() or not native.any():
        return np.asarray(grid, dtype=np.float32).copy()

    base = np.asarray(grid, dtype=np.float64).copy()
    rad = np.deg2rad(base)
    sin_g = np.sin(rad)
    cos_g = np.cos(rad)
    work = ocean & (native | fill_mask)

    for _ in range(n_iter):
        sin_mean = _neighbor_mean_4(sin_g, work)
        cos_mean = _neighbor_mean_4(cos_g, work)
        sin_g[fill_mask] = (1.0 - blend) * sin_g[fill_mask] + blend * sin_mean[fill_mask]
        cos_g[fill_mask] = (1.0 - blend) * cos_g[fill_mask] + blend * cos_mean[fill_mask]
        f_sin = sin_g[fill_mask]
        f_cos = cos_g[fill_mask]
        norm = np.hypot(f_sin, f_cos)
        ok = norm > 1e-8
        f_sin[ok] /= norm[ok]
        f_cos[ok] /= norm[ok]
        sin_g[fill_mask] = f_sin
        cos_g[fill_mask] = f_cos

    out = np.rad2deg(np.arctan2(sin_g, cos_g))
    out = (np.mod(out + 180.0, 360.0) - 180.0).astype(np.float32)
    out[native] = base[native].astype(np.float32)
    out[~ocean] = base[~ocean].astype(np.float32)
    return out


def high_latitude_ocean_mask(
    lats: np.ndarray,
    shape: tuple[int, int],
    *,
    threshold_deg: float = DEFAULT_POLAR_LAT_THRESHOLD_DEG,
) -> np.ndarray:
    """True where |latitude| >= *threshold_deg* (2-D, south-up grid)."""
    lat_2d = np.broadcast_to(np.asarray(lats)[:, None], shape)
    return np.abs(lat_2d) >= float(threshold_deg)


def fill_high_latitude_gaps_idw(
    grid: np.ndarray,
    gap: np.ndarray,
    ocean: np.ndarray,
    lats: np.ndarray,
    *,
    threshold_deg: float = DEFAULT_POLAR_LAT_THRESHOLD_DEG,
    circular: bool = False,
    write_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Fill high-latitude gap cells with IDW/Laplacian diffusion from native ocean cells.

    Low-latitude gap cells are left unchanged (e.g. after NN imputation).  Seeds
    are native observations only — imputed mid-latitude values are not used as
    sources, so polar fills extrapolate from the nearest real measurements.

    If *write_mask* is given, only those cells are written (still intersected with
    high-latitude ocean).  *gap* still defines native-seed exclusion, so already
    filled imputed cells (e.g. Arctic CHL POC) stay out of the seed set without
    being overwritten.
    """
    high_lat = high_latitude_ocean_mask(lats, ocean.shape, threshold_deg=threshold_deg)
    if write_mask is None:
        polar_gap = gap & ocean & high_lat
    else:
        polar_gap = np.asarray(write_mask, dtype=bool) & ocean & high_lat
    n_polar = int(polar_gap.sum())
    if n_polar == 0:
        return np.asarray(grid, dtype=np.float32).copy(), 0

    native = (~gap) & ocean & np.isfinite(grid)
    seeded = np.asarray(grid, dtype=np.float64).copy()
    # Hide all flagged gap cells so IDW seeds only from native observations.
    seeded[gap & ocean] = np.nan
    idw = fill_ocean_gaps(
        seeded,
        polar_gap,
        ocean,
        method="idw",
        circular=circular,
    )
    out = np.asarray(grid, dtype=np.float32).copy()
    out[polar_gap] = idw[polar_gap].astype(np.float32)
    out[native] = grid[native].astype(np.float32)
    out[~ocean] = grid[~ocean].astype(np.float32)
    return out, n_polar


def fill_ocean_gaps(
    grid: np.ndarray,
    gap: np.ndarray,
    ocean: np.ndarray,
    *,
    method: str = "zero",
    circular: bool = False,
) -> np.ndarray:
    """Fill ocean gap cells.

    method:
      - ``zero``: legacy 0-fill
      - ``nearest`` / ``nn``: nearest native ocean cell
      - ``idw`` / ``smooth``: nearest seed + Laplacian refinement (wave height/period)
    """
    out = np.asarray(grid, dtype=np.float64).copy()
    fill_mask = gap & ocean
    if not fill_mask.any():
        return out

    rem_nan = ocean & np.isnan(out)
    if rem_nan.any():
        out[rem_nan] = 0.0
        fill_mask = fill_mask | rem_nan

    if method == "zero":
        out[fill_mask] = 0.0
        return out

    valid = (~gap) & ocean & np.isfinite(grid)
    if not valid.any():
        out[fill_mask] = 0.0
        return out

    if circular:
        filled = _circular_nearest_fill(out, fill_mask, valid)
    elif method in {"nearest", "nn"}:
        valid_full = valid.copy()
        seeded = out.copy()
        seeded[~valid_full] = 0.0
        filled = _nearest_from_valid(seeded, valid_full)
        filled[valid] = out[valid]
    elif method in {"idw", "smooth", "laplacian"}:
        valid_full = valid.copy()
        seeded = out.copy()
        seeded[~valid_full] = 0.0
        nn = _nearest_from_valid(seeded, valid_full)
        nn[valid] = out[valid]
        filled = _laplacian_refine(nn, fill_mask, valid_full)
    else:
        raise ValueError(f"Unknown fill method: {method}")

    filled[~ocean] = out[~ocean]
    filled[valid] = out[valid]
    return filled.astype(np.float32)
