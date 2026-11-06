"""Shared utilities for feature-version comparison, install, and r50km scripts."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[1]
FEATURE_STACK = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
CACHE_DIR = ROOT / "data/raw/features/feature_updates"
DEFAULT_OUT = ROOT / "results/feature_version_comparison"
ETOPO_REF = FEATURE_STACK / "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd"
GRID_SHAPE = (2160, 4320)
R50_RADIUS_KM = 50.0


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def orient_map(grid: np.ndarray) -> np.ndarray:
    return np.rot90(np.rot90(np.fliplr(grid)))


def download(url: str, dest: Path, skip: bool) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if skip:
        raise FileNotFoundError(f"Missing cached file: {dest}")
    try:
        import requests
    except ImportError as exc:
        raise SystemExit("pip install requests") from exc
    print(f"Downloading\n  {url}\n  → {dest}")
    with requests.get(url, stream=True, timeout=3600) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
    return dest


def model_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(ETOPO_REF)
    lats = np.asarray(ds["lat"].values, dtype=np.float64)
    lons = np.asarray(ds["lon"].values, dtype=np.float64)
    elev = np.asarray(ds["data"].values, dtype=np.float64)
    ds.close()
    return lats, lons, elev


def load_old_grid(path: Path) -> np.ndarray:
    ds = xr.open_dataset(path)
    grid = np.asarray(ds["data"].values, dtype=np.float64)
    ds.close()
    return grid


def write_regridded(grid: np.ndarray, lats: np.ndarray, lons: np.ndarray, out_nc: Path) -> None:
    """Write a regridded comparison product (e.g. regridded_new/*_new_on_5m.nc)."""
    write_stack_grid(grid, out_nc, lats=xr.DataArray(lats), lons=xr.DataArray(lons))


def load_stack_grid(path: Path) -> tuple[np.ndarray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Load the main (2160, 4320) variable from a feature-stack NetCDF."""
    from mask_gap_core import fix_porosity_longitude, needs_porosity_longitude_flip

    ds = xr.open_dataset(path)
    attrs = dict(ds.attrs)
    var = None
    for name in ds.data_vars:
        if ds[name].ndim == 2 and tuple(ds[name].shape) == GRID_SHAPE:
            var = name
            break
    if var is None:
        ds.close()
        raise ValueError(f"No {GRID_SHAPE} data variable in {path}")
    data = ds[var]
    lats, lons = ds["lat"], ds["lon"]
    grid = np.asarray(data.values, dtype=np.float64)
    ds.close()
    if needs_porosity_longitude_flip(path, attrs):
        grid = fix_porosity_longitude(grid)
    return grid, data, lats, lons


def write_stack_grid(
    grid: np.ndarray,
    dest: Path,
    *,
    lats: xr.DataArray | np.ndarray | None = None,
    lons: xr.DataArray | np.ndarray | None = None,
    template: xr.DataArray | None = None,
) -> int:
    """Write a float32 ``data`` layer to the feature stack. Returns finite cell count."""
    arr = np.asarray(grid, dtype=np.float32)
    if arr.shape != GRID_SHAPE:
        raise ValueError(f"Expected {GRID_SHAPE}, got {arr.shape} for {dest}")

    if template is not None:
        coords = dict(template.coords)
        if lats is not None:
            coords["lat"] = lats
        if lons is not None:
            coords["lon"] = lons
        da = xr.DataArray(
            arr,
            dims=template.dims,
            coords=coords,
            name="data",
        )
        if hasattr(template, "attrs"):
            da.attrs = dict(template.attrs)
    else:
        if lats is None or lons is None:
            raise ValueError("write_stack_grid requires lats/lons or a template DataArray")
        lat_c = np.asarray(lats, dtype=np.float64)
        lon_c = np.asarray(lons, dtype=np.float64)
        da = xr.DataArray(
            arr,
            dims=("lat", "lon"),
            coords={"lat": ("lat", lat_c), "lon": ("lon", lon_c)},
            name="data",
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    ds = xr.Dataset({"data": da}, coords={"lat": da.coords.get("lat"), "lon": da.coords.get("lon")})
    if dest.exists():
        with xr.open_dataset(dest) as prior:
            ds.attrs.update(dict(prior.attrs))
    ds.to_netcdf(tmp)
    ds.close()
    tmp.replace(dest)
    return int(np.isfinite(arr).sum())


def install_grid_from_nc(src: Path, dest: Path) -> int:
    """Copy a ``data`` layer from a regridded comparison NetCDF into the feature stack."""
    ds = xr.open_dataset(src)
    if "data" not in ds:
        ds.close()
        raise ValueError(f"{src} missing 'data' variable")
    grid = np.asarray(ds["data"].values, dtype=np.float32)
    lats, lons = ds["lat"], ds["lon"]
    ds.close()
    return write_stack_grid(grid, dest, lats=lats, lons=lons)


def fill_profiles_along_depth(profiles: np.ndarray) -> np.ndarray:
    n_depth, n_lat, n_lon = profiles.shape
    flat = profiles.reshape(n_depth, -1)
    filled = pd.DataFrame(flat).bfill().ffill().to_numpy(dtype=np.float64).reshape(n_depth, n_lat, n_lon)
    return filled


def sample_at_seafloor(depths: np.ndarray, profiles: np.ndarray, seafloor_depth_m: np.ndarray) -> np.ndarray:
    profiles = fill_profiles_along_depth(profiles)
    n_depth, n_lat, n_lon = profiles.shape
    flat_depth = np.clip(seafloor_depth_m.ravel(), depths[0], depths[-1])
    prof = profiles.reshape(n_depth, -1)
    n_cells = prof.shape[1]
    out = np.full(n_cells, np.nan, dtype=np.float64)

    valid = np.isfinite(seafloor_depth_m.ravel()) & (seafloor_depth_m.ravel() > 0)
    if not np.any(valid):
        return out.reshape(n_lat, n_lon)

    td = flat_depth[valid]
    idx = np.searchsorted(depths, td, side="right") - 1
    idx = np.clip(idx, 0, len(depths) - 2)
    d0 = depths[idx]
    d1 = depths[idx + 1]
    w = (td - d0) / np.maximum(d1 - d0, 1e-12)
    cols = np.where(valid)[0]
    v0 = prof[idx, cols]
    v1 = prof[idx + 1, cols]
    out[cols] = v0 * (1.0 - w) + v1 * w
    return out.reshape(n_lat, n_lon)


def r50_filter_size(lat: xr.DataArray) -> int:
    deg_per_pixel = float(lat[1] - lat[0])
    km_per_pixel = deg_per_pixel * 111.0
    radius_px = int(np.round(R50_RADIUS_KM / km_per_pixel))
    return max(1, 2 * radius_px + 1)


def smooth_arithmetic_r50(
    grid: np.ndarray, size: int, *, invalid: np.ndarray | None = None
) -> np.ndarray:
    valid = np.isfinite(grid)
    if invalid is not None:
        valid = valid & ~invalid
    filled = np.where(valid, grid, 0.0)
    num = uniform_filter(filled, size=size, mode="constant", cval=0.0)
    den = uniform_filter(valid.astype(np.float64), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den <= 0] = np.nan
    return np.where(valid, out, np.nan)


def smooth_circular_deg_r50(grid: np.ndarray, size: int) -> np.ndarray:
    valid = np.isfinite(grid)
    rad = np.deg2rad(np.where(valid, grid, 0.0))
    sin_v = np.where(valid, np.sin(rad), 0.0)
    cos_v = np.where(valid, np.cos(rad), 0.0)
    sin_s = uniform_filter(sin_v, size=size, mode="constant", cval=0.0)
    cos_s = uniform_filter(cos_v, size=size, mode="constant", cval=0.0)
    wt = uniform_filter(valid.astype(np.float64), size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.rad2deg(np.arctan2(sin_s / wt, cos_s / wt))
    out[wt <= 0] = np.nan
    return np.where(valid, out, np.nan)


def build_r50km_companion(
    base_name: str,
    r50_name: str,
    *,
    circular: bool = False,
    stack_dir: Path | None = None,
    invalid_mask: np.ndarray | None = None,
    zero_invalid: bool = False,
) -> int:
    """Build one r50km companion grid from an installed base layer in the feature stack."""
    stack = stack_dir or FEATURE_STACK
    base_path = stack / base_name
    out_path = stack / r50_name
    if not base_path.exists():
        raise FileNotFoundError(base_path)

    grid, template, lats, lons = load_stack_grid(base_path)
    size = r50_filter_size(lats)
    if circular:
        smoothed = smooth_circular_deg_r50(grid, size)
    else:
        smoothed = smooth_arithmetic_r50(grid, size, invalid=invalid_mask)
    if zero_invalid and invalid_mask is not None:
        smoothed = np.where(invalid_mask, 0.0, smoothed)

    finite = write_stack_grid(smoothed, out_path, template=template, lats=lats, lons=lons)
    print(f"  wrote {out_path.name}  filter={size}px  finite={finite:,}")
    return finite


def normalize_lonlat_coords(da: xr.DataArray) -> xr.DataArray:
    """Map ERDDAP-style latitude/longitude coords to lat/lon."""
    rename = {}
    if "latitude" in da.coords and "lat" not in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords and "lon" not in da.coords:
        rename["longitude"] = "lon"
    return da.rename(rename) if rename else da


def validate_netcdf_path(path: Path, *, min_bytes: int = 1000) -> None:
    """Raise ValueError if path is missing, tiny, or an ERDDAP/HTML error body."""
    if not path.exists():
        raise FileNotFoundError(path)
    if path.stat().st_size < min_bytes:
        raise ValueError(f"{path} is too small ({path.stat().st_size} B) — incomplete download")
    head = path.read_bytes()[:64].lstrip()
    if head.startswith((b"<!", b"<html", b"<HTML", b"Error", b"error")):
        raise ValueError(f"{path} is an error page, not NetCDF")


def interp_to_model(
    src: xr.DataArray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    da = normalize_lonlat_coords(src)
    if float(da.lon.max()) > 180:
        da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby("lon")
    out = da.interp(lat=target_lat, lon=target_lon, method=method)
    return np.asarray(out.values, dtype=np.float64)


def apply_ocean_mask(grid: np.ndarray, elev: np.ndarray, ocean_only: bool) -> np.ndarray:
    if not ocean_only:
        return grid.astype(np.float64, copy=True)
    out = grid.astype(np.float64, copy=True)
    out[elev >= 0] = np.nan
    return out


def direction_pm180(deg: np.ndarray) -> np.ndarray:
    """Map wave direction to (-180, 180] degrees, matching the legacy WW3 stack."""
    out = deg.astype(np.float64, copy=True)
    out[~np.isfinite(out)] = np.nan
    return (np.mod(out + 180.0, 360.0) - 180.0).astype(np.float64)


def circular_mean_deg(da: xr.DataArray, dim: str = "time") -> xr.DataArray:
    """Vector (circular) mean of angles in degrees along *dim*."""
    rad = np.deg2rad(da)
    sin_m = np.sin(rad).mean(dim, skipna=True)
    cos_m = np.cos(rad).mean(dim, skipna=True)
    return xr.DataArray(
        np.rad2deg(np.arctan2(sin_m.values, cos_m.values)),
        coords=sin_m.coords,
        dims=sin_m.dims,
    )


def circular_diff_deg(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    """Signed shortest angular difference new − old in degrees."""
    return (np.mod(new - old + 180.0, 360.0) - 180.0).astype(np.float64)


def plot_triptych(
    old: np.ndarray,
    new: np.ndarray,
    *,
    title: str,
    unit: str,
    old_label: str,
    new_label: str,
    out_path: Path,
    elev: np.ndarray,
    ocean_only: bool = False,
    circular_deg: bool = False,
) -> dict[str, float]:
    old_m = apply_ocean_mask(old, elev, ocean_only)
    new_m = apply_ocean_mask(new, elev, ocean_only)
    if circular_deg:
        old_m = direction_pm180(old_m)
        new_m = direction_pm180(new_m)
        diff = circular_diff_deg(new_m, old_m)
        plot_unit = "deg (−180…180, from north)"
    else:
        diff = new_m - old_m
        plot_unit = unit

    old_p = orient_map(old_m)
    new_p = orient_map(new_m)
    diff_p = orient_map(diff)

    finite_old = old_p[np.isfinite(old_p)]
    finite_new = new_p[np.isfinite(new_p)]
    finite_diff = diff_p[np.isfinite(diff_p)]

    if circular_deg:
        old_p = np.mod(old_p, 360.0)
        new_p = np.mod(new_p, 360.0)
        vmin, vmax = 0.0, 360.0
        map_cmap = "hsv"
    else:
        vmin = float(np.nanpercentile(finite_old, 2)) if finite_old.size else 0.0
        vmax = float(np.nanpercentile(finite_old, 98)) if finite_old.size else 1.0
        if vmin == vmax and finite_old.size:
            vmin, vmax = float(np.nanmin(finite_old)), float(np.nanmax(finite_old))
        if finite_new.size and not circular_deg:
            vmin = min(vmin, float(np.nanpercentile(finite_new, 2)))
            vmax = max(vmax, float(np.nanpercentile(finite_new, 98)))
        map_cmap = "viridis"

    dlim = float(np.nanpercentile(np.abs(finite_diff), 98)) if finite_diff.size else 1.0
    if dlim == 0:
        dlim = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    extent = [-180, 180, -90, 90]
    diff_title = "Circular diff (new − old)" if circular_deg else "Difference (new − old)"
    panels = [
        (old_p, f"Old: {old_label}", vmin, vmax, map_cmap),
        (new_p, f"New: {new_label}", vmin, vmax, map_cmap),
        (diff_p, diff_title, -dlim, dlim, "RdBu_r"),
    ]
    for ax, (data, ptitle, lo, hi, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            im = ax.imshow(data, cmap=cmap, norm=norm, extent=extent, origin="upper")
        else:
            im = ax.imshow(data, cmap=cmap, vmin=lo, vmax=hi, extent=extent, origin="upper")
        ax.set_title(f"{title}\n{ptitle}", fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=plot_unit)

    fig.suptitle(f"{title} — same 5′ grid", fontsize=11, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    stats: dict[str, float] = {}
    both = np.isfinite(old_m) & np.isfinite(new_m)
    if both.any():
        d = diff[both]
        stats["rmse"] = float(np.sqrt(np.mean(d**2)))
        stats["mean_bias"] = float(np.mean(d))
        stats["median_bias"] = float(np.median(d))
        stats["pct_95_abs_diff"] = float(np.percentile(np.abs(d), 95))
        stats["n_cells"] = int(both.sum())
    return stats


def log10_safe(grid: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """log10 of positive values (used where the stack stores log10)."""
    g = grid.astype(np.float64, copy=True)
    g[~np.isfinite(g)] = np.nan
    g[g <= 0] = np.nan
    return np.log10(np.maximum(g, floor))


def ln_safe(grid: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Natural log of positive values."""
    g = grid.astype(np.float64, copy=True)
    g[~np.isfinite(g)] = np.nan
    g[g <= 0] = np.nan
    return np.log(np.maximum(g, floor))
