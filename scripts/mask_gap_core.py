"""Shared ocean-gap detection for harmonized masked features."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
# Optional native-evidence trees (used only for mask diagnostics / partial coverage).
# Override with env FEATURES_PARTIAL_DIR / EOT20_PARTIAL_DIR when available.
_DEFAULT_PARTIAL = Path("/gxfs_work/cau/sunms498/work_old/Data/Features_new")
_DEFAULT_EOT20 = Path(
    "/gxfs_work/cau/sunms498/work_old/Data/tides/ocean_tides/ocean_tides/derived_eot20_stats"
)
PARTIAL = Path(os.environ.get("FEATURES_PARTIAL_DIR", str(_DEFAULT_PARTIAL)))
NEW_DIR = ROOT / "results/feature_version_comparison/regridded_new"
META = ROOT / "data/interim/selectedFeatureLists/extrapolated_features_metadata.csv"
EOT20_PARTIAL = Path(os.environ.get("EOT20_PARTIAL_DIR", str(_DEFAULT_EOT20)))
OUT_MASKS = ROOT / "data/interim/masks"
OCEAN_GAP_STACK = OUT_MASKS / "ocean_gap_stack.npy"
OCEAN_GAP_NAMES = OUT_MASKS / "ocean_gap_feature_names.txt"
GARLAN_GAP_CACHE = OUT_MASKS / "garlan_unclassified_ocean_gap.npy"
INTERP_MASK_STACK = OUT_MASKS / "interpolation_mask_stack.npy"
INTERP_MASK_NAMES = OUT_MASKS / "interpolation_mask_feature_names.txt"

# regridded_new *_new_on_5m.nc -> stack filename
RAW_NEW_MAP: dict[str, str] = {
    "elevation_new_on_5m.nc": "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd",
    "sediment_thickness_new_on_5m.nc": "GL_TOT_SED_THICK_M_GLOBSED_v3.5m.nc",
    "crust_age_new_on_5m.nc": "SC_CRUST_AGE_MA_Muller2022.5m.nc",
    "temperature_new_on_5m.nc": "SF_SEA_TEMPERATURE_C_DECADAL_MEAN_woa23x.5m.grd",
    "salinity_new_on_5m.nc": "SF_SEA_SALINITY_PSU_DECADAL_MEAN_woa23x.5m.nc",
    "oxygen_new_on_5m.nc": "SF_SEA_OXYGEN_MLL_DECADAL_MEAN_woa23x.5m.grd",
    "oxygen_pctsat_new_on_5m.nc": "SF_SEA_OXYGEN_PCTSAT_DECADAL_MEAN_woa23x.5m.grd",
    "aou_new_on_5m.nc": "SF_SEA_SEA_OXYGEN_UTILIZATION_MOLM3_DECADAL_MEAN_woa23x.5m.grd",
    "river_co2_new_on_5m.nc": "GL_RIVERMOUTH_CO2_TGCYR-1_Liu2024.5m.grd",
    "river_doc_new_on_5m.nc": "GL_RIVERMOUTH_DOC_TGCYR-1_Liu2024.5m.grd",
    "river_hco3_new_on_5m.nc": "GL_RIVERMOUTH_HCO3_TGCYR-1_Liu2024.5m.grd",
    "river_poc_new_on_5m.nc": "GL_RIVERMOUTH_POC_TGCYR-1_Liu2024.5m.grd",
    "mimoc_mld_max_new_on_5m.nc": "SS_MIXED_LAYER_DEPTH_MAX_M_MIMOCx.5m.grd",
    "mimoc_mld_min_new_on_5m.nc": "SS_MIXED_LAYER_DEPTH_MIN_M_MIMOCx.5m.grd",
    "viirs_chlorophyll_new_on_5m.nc": "SS_CHLOROPHYLL_LOG_MG_M3_VIIRS_SNPP_MISSION_MEANx.5m.grd",
    "viirs_pic_new_on_5m.nc": "SS_PIC_LOG_MOL_M3-1_VIIRS_SNPP_MISSION_MEANx.5m.grd",
    "viirs_poc_new_on_5m.nc": "SS_POC_LOG_MOL_M3-1_VIIRS_SNPP_MISSION_MEANx.5m.grd",
    "woa23_sss_density_new_on_5m.nc": "SS_DENSITY_KGM-3_woa23x.5m.grd",
    "ww3_wave_height_new_on_5m.nc": "SS_WAVE_HEIGHT_M_2018_2023_WAVEWATCH3x.5m.grd",
    "ww3_wave_period_new_on_5m.nc": "SS_WAVE_PERIOD_S_2018_2023_WAVEWATCH3x.5m.grd",
    "ww3_wave_direction_new_on_5m.nc": "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd",
    "era5_wind_new_on_5m.nc": "SS_WINDSPEED_MS-1_ERA5.5m.grd",
    "hycom_u_east_new_on_5m.nc": "SF_CURRENT_EAST_MS_2018_2023_HYCOMx.5m.grd",
    "hycom_u_north_new_on_5m.nc": "SF_CURRENT_NORTH_MS_2018_2023_HYCOMx.5m.grd",
    "hycom_u_mag_new_on_5m.nc": "SF_CURRENT_MAG_MS_2018_2023_HYCOMx.5m.grd",
}

ARCTIC_POC_BASE_NC = NEW_DIR / "viirs_poc_base_new_on_5m.nc"

WAVE_BASE_FEATURES = frozenset(
    {
        "SS_WAVE_HEIGHT_M_2018_2023_WAVEWATCH3x.5m.grd",
        "SS_WAVE_PERIOD_S_2018_2023_WAVEWATCH3x.5m.grd",
        "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd",
    }
)

# Wave direction is encoded for the model as sin/cos component channels instead
# of raw degrees (removes the ±180° branch cut). The component grids are derived
# from the already gap-filled base direction grid, so they inherit its gap mask.
WAVE_DIRECTION_BASE_DEG = "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd"
WAVE_DIRECTION_COMPONENTS = frozenset(
    {
        "SS_WAVE_DIRECTION_SIN_2018_2023_WAVEWATCH3x.5m.nc",
        "SS_WAVE_DIRECTION_COS_2018_2023_WAVEWATCH3x.5m.nc",
        "SS_WAVE_DIRECTION_SIN_2018_2023_WAVEWATCH3x.r50km.men.5m.nc",
        "SS_WAVE_DIRECTION_COS_2018_2023_WAVEWATCH3x.r50km.men.5m.nc",
    }
)


def is_wave_base_feature(fname: str) -> bool:
    return fname in WAVE_BASE_FEATURES


def is_wave_direction_component(fname: str) -> bool:
    return fname in WAVE_DIRECTION_COMPONENTS


def is_wave_feature(fname: str) -> bool:
    return fname.startswith("SS_WAVE_")


STACK_TO_RAW: dict[str, Path] = {
    stack: NEW_DIR / raw for raw, stack in RAW_NEW_MAP.items() if (NEW_DIR / raw).exists()
}

CRUST_AGE_FEATURES = {
    "SC_CRUST_AGE_MA_Muller2022.5m.nc",
    "SC_CRUST_AGE_MA_Muller2022.5m_r50km.nc",
}

GARLAN_FEATURES = {
    "DERIVATIVE_lithology_grain_size_Everardo_global_8.nc",
    "litho_maps_type1_.nc",
    "litho_maps_type2_.nc",
    "litho_maps_type3_.nc",
    "litho_maps_type4_.nc",
    "litho_maps_type5_.nc",
    "litho_maps_type6_.nc",
    "lithology_grain_size_Everardo_global_8.nc",
}

R50_REPLACEMENTS = [
    (".r50km.men.5m.grd", ".5m.grd"),
    (".r50km.men.5m.grd", ".5m.nc"),
    (".r50km.men.5m.nc", ".5m.nc"),
    ("_r50km.nc", ".nc"),
    ("_r50km_men.nc", ".nc"),
    ("_r50km.5m.nc", ".5m.nc"),
]

RIVERMOUTH_PREFIX = "GL_RIVERMOUTH_"

# Legacy FINAL-stack porosity layers were stored E/W mirrored vs ETOPO.
# Fixed stack files carry attrs[porosity_longitude_fixed] and are not flipped.
# Features_new "*_with_coords_flip.nc" is already E/W-aligned to ETOPO — do not
# flip again (that invented ~1.3M false ocean gaps / a reversed native map).
POROSITY_LONGITUDE_FLIP_FILES = frozenset(
    {
        "SF_POROSITY_FRC.5m_with_coords_flip.nc",
        "SF_POROSITY_SDEV_FRC.5m_with_coords_flip.nc",
    }
)
POROSITY_LON_FIXED_ATTR = "porosity_longitude_fixed"


def needs_porosity_longitude_flip(path: Path, attrs: dict | None = None) -> bool:
    if path.name not in POROSITY_LONGITUDE_FLIP_FILES:
        return False
    if attrs and attrs.get(POROSITY_LON_FIXED_ATTR):
        return False
    # Archived pre-fill snapshot is already in the flipped/ETOPO convention.
    if "Features_new" in Path(path).parts:
        return False
    return True


def fix_porosity_longitude(arr: np.ndarray) -> np.ndarray:
    """Mirror porosity along longitude to match ETOPO / island_map convention."""
    return np.asarray(arr)[:, ::-1].copy()


def is_rivermouth_feature(fname: str) -> bool:
    """River-mouth flux layers: open-ocean zero is physical, not gap-filled."""
    return fname.startswith(RIVERMOUTH_PREFIX)


def is_garlan_feature(fname: str) -> bool:
    """Garlan lithology layers: binary type maps and grain-size derivatives."""
    return fname in GARLAN_FEATURES


def training_feature_names() -> list[str]:
    names = [
        ln.strip()
        for ln in (ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt")
        .read_text()
        .splitlines()
        if ln.strip()
    ]
    # The first list entry is a non-training reference column (elevation
    # derivative) that is always dropped; guard by name so the count can change.
    if names and names[0].startswith("DERIVATIVE_GL_ELEVATION"):
        return names[1:]
    return names


def resolve_r50_base(fname: str, known: set[str]) -> str | None:
    for old, new in R50_REPLACEMENTS:
        if old in fname:
            base = fname.replace(old, new)
            if base in known:
                return base
    return None


def load_grid(path: Path) -> np.ndarray:
    arr, _, _ = load_grid_with_coords(path)
    return arr


def load_grid_with_coords(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    attrs = dict(ds.attrs)
    if "data" in ds:
        arr = ds["data"].values
    else:
        for v in ds.data_vars:
            if ds[v].ndim >= 2:
                arr = ds[v].values
                break
        else:
            arr = ds[list(ds.data_vars)[0]].values
    lats = np.asarray(ds["lat"].values)
    lons = np.asarray(ds["lon"].values)
    ds.close()
    arr = np.asarray(arr)
    if needs_porosity_longitude_flip(path, attrs):
        arr = fix_porosity_longitude(arr)
    return arr, lats, lons


def align_to_reference_grid(
    arr: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> np.ndarray:
    """Flip array axes when archived partial uses descending lat/lon."""
    src_lat = np.atleast_1d(np.asarray(src_lat))
    src_lon = np.atleast_1d(np.asarray(src_lon))
    if arr.shape != (len(ref_lat), len(ref_lon)):
        return arr
    out = arr
    if src_lat.size > 1 and src_lat[0] > src_lat[-1]:
        out = out[::-1, :]
    if src_lon.size > 1 and src_lon[0] > src_lon[-1]:
        out = out[:, ::-1]
    return out


def load_meta_map(meta_path: Path) -> dict[str, tuple[re.Pattern | None, dict[str, str]]]:
    meta_map: dict[str, tuple[re.Pattern | None, dict[str, str]]] = {}
    for row in csv.DictReader(meta_path.open()):
        pat = row["feature_file"]
        if "*" in pat:
            rx = re.compile("^" + re.escape(pat).replace(r"\*", ".*") + "$")
            meta_map[pat] = (rx, row)
        else:
            meta_map[pat] = (None, row)
    return meta_map


def empirical_gap(
    fname: str,
    arr_final: np.ndarray,
    ocean: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> np.ndarray | None:
    partial_path = PARTIAL / fname
    if not partial_path.exists() or arr_final.shape != ocean.shape:
        return None
    arr_partial, partial_lat, partial_lon = load_grid_with_coords(partial_path)
    if arr_partial.shape != arr_final.shape:
        return None
    arr_partial = align_to_reference_grid(arr_partial, partial_lat, partial_lon, ref_lat, ref_lon)
    return ocean & np.isnan(arr_partial) & np.isfinite(arr_final)


def eot20_gap(fname: str, arr_final: np.ndarray, ocean: np.ndarray) -> np.ndarray | None:
    if not fname.startswith("EOT20_") or not fname.endswith("_interp_land.nc"):
        return None
    base_name = fname.replace("_interp_land.nc", ".nc")
    partial_path = EOT20_PARTIAL / base_name
    if not partial_path.exists():
        return None
    arr_partial = load_grid(partial_path)
    if arr_partial.shape != arr_final.shape:
        return None
    return ocean & np.isnan(arr_partial) & np.isfinite(arr_final)


def jorgenson_gap(fname: str, arr_final: np.ndarray, ocean: np.ndarray) -> np.ndarray | None:
    if fname != "TOU_Jorgenson2022.nc":
        return None
    return ocean & (arr_final < 0)


def muller_gap(fname: str, ocean: np.ndarray) -> np.ndarray | None:
    if fname not in CRUST_AGE_FEATURES:
        return None
    cache = OUT_MASKS / "crust_age_ocean_gap.npy"
    if cache.exists():
        g = np.load(cache).astype(bool)
        if g.shape == ocean.shape:
            return g
    raw = STACK_TO_RAW.get("SC_CRUST_AGE_MA_Muller2022.5m.nc")
    if raw and raw.exists():
        raw_arr = load_grid(raw)
        if raw_arr.shape == ocean.shape:
            return ocean & np.isnan(raw_arr)
    return None


def garlan_gap(fname: str, ocean: np.ndarray, garlan_unclassified: np.ndarray) -> np.ndarray | None:
    if fname not in GARLAN_FEATURES:
        return None
    return garlan_unclassified.astype(bool)


def garlan_gap_mask(ocean: np.ndarray) -> np.ndarray:
    """Shared gap mask for all Garlan lithology layers."""
    return build_garlan_unclassified_ocean_mask(ocean)


def arctic_chl_poc_gap(fname: str, arr_final: np.ndarray, ocean: np.ndarray) -> np.ndarray | None:
    """POC cells filled from Copernicus Arctic CHL regression (not VIIRS native)."""
    if fname != "SS_POC_LOG_MOL_M3-1_VIIRS_SNPP_MISSION_MEANx.5m.grd":
        return None
    if not ARCTIC_POC_BASE_NC.exists():
        return None
    arr_base = load_grid(ARCTIC_POC_BASE_NC)
    if arr_base.shape != arr_final.shape:
        return None
    return ocean & np.isnan(arr_base) & np.isfinite(arr_final)


def raw_source_gap(fname: str, ocean: np.ndarray) -> np.ndarray | None:
    raw_path = STACK_TO_RAW.get(fname)
    if raw_path is None or not raw_path.exists():
        return None
    raw = load_grid(raw_path)
    if raw.shape != ocean.shape:
        return None
    return ocean & np.isnan(raw)


def stack_nan_gap(arr: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    return ocean & np.isnan(arr)


def build_garlan_unclassified_ocean_mask(ocean: np.ndarray) -> np.ndarray:
    """Ocean cells with no Garlan lithology class (CarteSedimMonde unclassified).

    Uses a persistent cache when available so gap detection stays valid after
    gap-fill has written non-zero values into the lithology type layers.
    """
    if GARLAN_GAP_CACHE.exists():
        cached = np.load(GARLAN_GAP_CACHE)
        if cached.shape == ocean.shape:
            return (cached.astype(bool) & ocean)

    if INTERP_MASK_STACK.exists() and INTERP_MASK_NAMES.exists():
        names = [ln.strip() for ln in INTERP_MASK_NAMES.read_text().splitlines() if ln.strip()]
        if "litho_maps_type1_.nc" in names:
            stack = np.load(INTERP_MASK_STACK)
            idx = names.index("litho_maps_type1_.nc")
            gap = stack[:, :, idx].astype(bool) & ocean
            GARLAN_GAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.save(GARLAN_GAP_CACHE, gap.astype(np.uint8))
            return gap

    sum_types = np.zeros(ocean.shape, dtype=np.float64)
    for type_idx in range(1, 7):
        sum_types += load_grid(FINAL / f"litho_maps_type{type_idx}_.nc")
    gap = (ocean & (sum_types == 0)).astype(bool)
    GARLAN_GAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(GARLAN_GAP_CACHE, gap.astype(np.uint8))
    return gap


def rivermouth_offmouth_zero_mask(
    fname: str, ocean: np.ndarray
) -> tuple[np.ndarray, str] | None:
    """Do not flag open-ocean zeros as interpolated (no river discharge off-mouth)."""
    if not is_rivermouth_feature(fname):
        return None
    return np.zeros_like(ocean, dtype=bool), "rivermouth_offmouth_zero"


def detect_ocean_gap(
    fname: str,
    arr: np.ndarray,
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    garlan_unclassified: np.ndarray,
    base_gaps: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Return (gap bool array, source label). Priority: proven gaps, then cell NaN."""
    rivermouth = rivermouth_offmouth_zero_mask(fname, ocean)
    if rivermouth is not None:
        return rivermouth

    if is_wave_direction_component(fname):
        # Inherit the base direction gap (raw WW3 product NaN over ocean); all
        # sin/cos components (base + r50km) share the same imputed cells.
        parent_gap = raw_source_gap(WAVE_DIRECTION_BASE_DEG, ocean)
        if parent_gap is not None and parent_gap.any():
            return parent_gap.astype(bool), "wave_direction_component"
        return stack_nan_gap(arr, ocean), "stack_ocean_nan"

    known = set(base_gaps or {})
    base = resolve_r50_base(fname, known)
    if base is not None and base_gaps is not None and base in base_gaps:
        return base_gaps[base].copy(), "r50km_inherited"

    # POC: Arctic CHL-derived cells are already filled but must stay flagged as
    # imputed; remaining ocean NaNs also need to be in the gap mask or NN/IDW
    # never sees them (previous bug: arctic_chl_derived alone short-circuited).
    arctic = arctic_chl_poc_gap(fname, arr, ocean)
    if arctic is not None:
        nan = stack_nan_gap(arr, ocean)
        combined = arctic.astype(bool) | nan
        if combined.any():
            if nan.any() and arctic.any():
                return combined, "arctic_chl_derived+stack_ocean_nan"
            if arctic.any():
                return arctic.astype(bool), "arctic_chl_derived"
            return nan, "stack_ocean_nan"

    for detector, source in (
        (lambda: raw_source_gap(fname, ocean), "raw_product_nan"),
        (lambda: empirical_gap(fname, arr, ocean, lats, lons), "empirical_partial_to_final"),
        (lambda: eot20_gap(fname, arr, ocean), "eot20_polar_ocean_fill"),
        (lambda: jorgenson_gap(fname, arr, ocean), "jorgenson_ocean_gap"),
        (lambda: muller_gap(fname, ocean), "muller2022_ocean_gap"),
        (lambda: garlan_gap(fname, ocean, garlan_unclassified), "garlan_unclassified"),
        (lambda: stack_nan_gap(arr, ocean), "stack_ocean_nan"),
    ):
        gap = detector()
        if gap is not None and gap.any():
            return gap.astype(bool), source

    gap = stack_nan_gap(arr, ocean)
    if gap.any():
        return gap, "stack_ocean_nan"
    return np.zeros_like(ocean, dtype=bool), "none_observed"


def load_ocean_gap_stack() -> tuple[np.ndarray, list[str]] | None:
    if not OCEAN_GAP_STACK.exists() or not OCEAN_GAP_NAMES.exists():
        return None
    names = [ln.strip() for ln in OCEAN_GAP_NAMES.read_text().splitlines() if ln.strip()]
    stack = np.load(OCEAN_GAP_STACK)
    if stack.shape[-1] != len(names):
        return None
    return stack, names


def load_training_mask_stack() -> np.ndarray:
    """Gap indicators for the 113 training predictors (same source as diagnostic maps).

    Reads ``interpolation_mask_stack.npy`` — the authoritative mask written by
    ``build_interpolation_masks.py`` — and drops the non-training elevation-derivative
    column when present.  Training arrays, world cubes, and ``plot_masked_feature_maps``
    must all use this stack so z-scoring stats, imputed values, and mask channels agree.
    """
    if not INTERP_MASK_STACK.exists() or not INTERP_MASK_NAMES.exists():
        raise FileNotFoundError(
            f"Missing {INTERP_MASK_STACK} — run build_interpolation_masks.py first"
        )
    names = [ln.strip() for ln in INTERP_MASK_NAMES.read_text().splitlines() if ln.strip()]
    stack = np.load(INTERP_MASK_STACK)
    train_names = training_feature_names()
    if names and names[0].startswith("DERIVATIVE_GL_ELEVATION") and len(names) == len(train_names) + 1:
        stack = stack[:, :, 1:]
        names = names[1:]
    if stack.shape[-1] != len(train_names) or names != train_names:
        raise ValueError(
            f"Mask stack shape {stack.shape} / {len(names)} names does not match "
            f"{len(train_names)} training predictors"
        )
    return stack.astype(np.float32)


def load_reference_ocean_mask() -> np.ndarray:
    """ETOPO ocean mask (True = ocean) on the 2160×4320 model grid."""
    elev = load_grid(FINAL / "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd")
    return elev < 0


def gap_for_feature_name(fname: str) -> np.ndarray | None:
    loaded = load_ocean_gap_stack()
    if loaded is None:
        return None
    stack, names = loaded
    if fname not in names:
        return None
    return stack[:, :, names.index(fname)].astype(bool)
