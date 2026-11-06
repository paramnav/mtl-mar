#!/usr/bin/env python3
"""Build deterministic per-feature ocean interpolation indicator masks.

Mask definition (per feature f, ocean cell):
  1 = value was filled/interpolated on the ocean grid
  0 = observed/native on the ocean grid

Evidence tiers (in priority order):
  - empirical: NaN in Features_new partial but finite in final grid
  - eot20_polar: ocean cells filled when comparing non-interp vs interp_land EOT20
  - jorgenson_ocean_gap: ocean cells with sentinel TOU < 0 (unfilled Arctic/ice gaps)
  - garlan_unclassified: Garlan lithology — ocean cells with all type probabilities zero
  - harmonized_ocean_gap: saved cell-level gaps from harmonize_ocean_gaps.py
  - stack_ocean_nan: ocean NaN in current stack (cell-level, not metadata)
  - rivermouth_offmouth_zero: GL_RIVERMOUTH_* — open-ocean zero flux is physical
  - none: all zeros (treated as fully observed on ocean)

Outputs:
  data/interim/masks/interpolation_mask_stack.npy       (2160, 4320, n_features) uint8
  data/interim/masks/interpolation_mask_feature_names.txt
  data/interim/masks/interpolation_mask_summary.csv
  results/diesing_plots/interpolation_mask_<feature>.png  (empirical layers only)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from mask_gap_core import (  # noqa: E402
    detect_ocean_gap,
    gap_for_feature_name,
    build_garlan_unclassified_ocean_mask,
    is_rivermouth_feature,
    load_grid,
    load_grid_with_coords,
    training_feature_names,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "data/raw/features/FeaturesPhrampusLee_TOCSedRate_updated"
PARTIAL = Path(__import__("os").environ.get("FEATURES_PARTIAL_DIR", "/gxfs_work/cau/sunms498/work_old/Data/Features_new"))
SEL = ROOT / "data/interim/selectedFeatureLists/selectedfeatures_v_men.txt"
META = ROOT / "data/interim/selectedFeatureLists/extrapolated_features_metadata.csv"
EOT20_PARTIAL = Path(__import__("os").environ.get("EOT20_PARTIAL_DIR", "/gxfs_work/cau/sunms498/work_old/Data/tides/ocean_tides/ocean_tides/derived_eot20_stats"))
OUT_MASKS = ROOT / "data/interim/masks"
OUT_PLOTS = ROOT / "results/diesing_plots"
MULLER_RAW = ROOT / "results/feature_version_comparison/regridded_new/crust_age_new_on_5m.nc"
CRUST_GAP_CACHE = OUT_MASKS / "crust_age_ocean_gap.npy"

CRUST_AGE_FEATURES = {
    "SC_CRUST_AGE_MA_Muller2022.5m.nc",
    "SC_CRUST_AGE_MA_Muller2022.5m_r50km.nc",
}

EMPIRICAL_FEATURES = {
    "probabilities_Calc.Sed_mod.nc",
    "probabilities_Clay_mod.nc",
    "probabilities_Dia.Ooze_mod.nc",
    "probabilities_Lith.Sed_mod.nc",
    "ISOPACH_0_1.8_METERS_KNNPREDICTION.5m.nc",
    "SEAFLOOR_DENSITY_GCM3.nc",
    "SF_POROSITY_FRC.5m_with_coords_flip.nc",
    "GL_COMPACT_ALL_CLY_ISOPACH_SED_THICK_M_GLOBSED_1_8_Straume.1d_5_arc_min_interpolated.nc",
    "GL_COMPACT_ALL_SND_ISOPACH_SED_THICK_M_GLOBSED_1_8_Straume.1d_5_arc_min_interpolated.nc",
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


def align_to_reference_grid(
    arr: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> np.ndarray:
    """Flip array axes when archived partial uses descending lat/lon."""
    if arr.shape != (len(ref_lat), len(ref_lon)):
        return arr
    out = arr
    if src_lat[0] > src_lat[-1]:
        out = out[::-1, :]
    if src_lon[0] > src_lon[-1]:
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


def meta_for(fname: str, meta_map: dict) -> dict[str, str] | None:
    for pat, (rx, row) in meta_map.items():
        if rx is not None and rx.match(fname):
            return row
        if pat == fname:
            return row
    # Wildcard rows in metadata (e.g. SS_BIOMASS_*_WEI2010x.5m.grd)
    if fname.startswith("SS_BIOMASS_") and "WEI2010x" in fname:
        return {
            "extrapolation_type": "ML_IDW_from_sparse_samples",
            "notes": "Wei et al. biomass RF+IDW",
        }
    if fname.startswith("SF_SEA_") and ("woa13" in fname.lower() or "woa23" in fname.lower()):
        return {
            "extrapolation_type": "climatology_interpolation",
            "notes": "WOA13 upsampled",
        }
    if fname.startswith("GL_RIVERMOUTH_") and fname.endswith(".grd"):
        return {
            "extrapolation_type": "coastal_point_upsampled",
            "notes": "River mouth upsampling",
        }
    return None


def empirical_mask(
    fname: str,
    arr_final: np.ndarray,
    ocean: np.ndarray,
    partial_dir: Path,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> tuple[np.ndarray, str] | None:
    partial_path = partial_dir / fname
    if not partial_path.exists() or arr_final.shape != ocean.shape:
        return None
    arr_partial, partial_lat, partial_lon = load_grid_with_coords(partial_path)
    if arr_partial.shape != arr_final.shape:
        return None
    arr_partial = align_to_reference_grid(
        arr_partial, partial_lat, partial_lon, ref_lat, ref_lon
    )
    mask = (ocean & np.isnan(arr_partial) & np.isfinite(arr_final)).astype(np.uint8)
    return mask, "empirical_partial_to_final"


def garlan_unclassified_mask(
    fname: str,
    ocean: np.ndarray,
    garlan_unclassified: np.ndarray,
) -> tuple[np.ndarray, str] | None:
    if fname not in GARLAN_FEATURES:
        return None
    return garlan_unclassified.copy(), "garlan_unclassified"


def jorgenson_ocean_gap_mask(
    fname: str, arr_final: np.ndarray, ocean: np.ndarray
) -> tuple[np.ndarray, str] | None:
    """Mark unfilled sentinel TOU values on ocean (not real observations)."""
    if fname != "TOU_Jorgenson2022.nc":
        return None
    mask = (ocean & (arr_final < 0)).astype(np.uint8)
    return mask, "jorgenson_ocean_gap"


def load_crust_age_ocean_gap(ocean: np.ndarray) -> np.ndarray | None:
    """Ocean cells with no native Müller 2022 crust age (NaN in source product)."""
    if CRUST_GAP_CACHE.exists():
        gap = np.load(CRUST_GAP_CACHE)
        if gap.shape == ocean.shape:
            return gap.astype(bool)
    if MULLER_RAW.exists():
        raw = load_grid(MULLER_RAW)
        if raw.shape == ocean.shape:
            return ocean & np.isnan(raw)
    return None


def muller2022_ocean_gap_mask(
    fname: str, ocean: np.ndarray
) -> tuple[np.ndarray, str] | None:
    if fname not in CRUST_AGE_FEATURES:
        return None
    gap = load_crust_age_ocean_gap(ocean)
    if gap is None:
        return None
    return gap.astype(np.uint8), "muller2022_ocean_gap"


def eot20_polar_mask(fname: str, arr_final: np.ndarray, ocean: np.ndarray) -> tuple[np.ndarray, str] | None:
    if not fname.startswith("EOT20_") or not fname.endswith("_interp_land.nc"):
        return None
    base_name = fname.replace("_interp_land.nc", ".nc")
    partial_path = EOT20_PARTIAL / base_name
    if not partial_path.exists():
        return None
    arr_partial = load_grid(partial_path)
    if arr_partial.shape != arr_final.shape:
        return None
    mask = (ocean & np.isnan(arr_partial) & np.isfinite(arr_final)).astype(np.uint8)
    return mask, "eot20_polar_ocean_fill"


def metadata_mask(
    fname: str,
    extrap_type: str,
    elev: np.ndarray,
    ocean: np.ndarray,
    shelf: np.ndarray,
    arr_final: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Metadata fallback — only cell-level rules (no whole-ocean conservative mask)."""
    if extrap_type == "Diesing_shelf_nearest_deep_water":
        return shelf.astype(np.uint8), "metadata_diesing_shelf"
    if extrap_type == "young_crust_griddata":
        young = ocean & (load_grid(FINAL / "SC_CRUST_AGE_MA_Muller2022.5m.nc") <= 1.0)
        return young.astype(np.uint8), "metadata_young_crust"
    nan_gap = (ocean & np.isnan(arr_final)).astype(np.uint8)
    if nan_gap.any():
        return nan_gap, f"stack_ocean_nan:{extrap_type}"
    return np.zeros_like(elev, dtype=np.uint8), "none_observed"


def plot_mask(
    mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    title: str,
    out_path: Path,
    ocean: np.ndarray,
    shelf: np.ndarray,
    deep: np.ndarray,
) -> None:
    display = np.where(ocean, mask.astype(float), np.nan)
    n_ocean = int((mask & ocean).sum())
    n_shelf = int((mask & shelf).sum())
    n_deep = int((mask & deep).sum())
    stats = (
        f"Interpolated ocean cells: {n_ocean:,}  |  "
        f"shelf: {n_shelf:,}  |  deep: {n_deep:,}"
    )

    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    im = ax.imshow(
        display,
        origin="lower",
        extent=[lons.min(), lons.max(), lats.min(), lats.max()],
        cmap="Reds",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="1 = interpolated")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.text(
        0.01,
        0.02,
        stats,
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot-empirical-only",
        action="store_true",
        default=True,
        help="Plot PNG maps for the 9 empirically proven layers (default: on)",
    )
    args = parser.parse_args()

    feature_names = [ln.strip() for ln in SEL.read_text().splitlines() if ln.strip()]
    meta_map = load_meta_map(META)

    elev, lats, lons = load_grid_with_coords(FINAL / "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd")
    ocean = elev < 0
    shelf = ocean & (elev > -500)
    deep = ocean & (elev <= -500)
    garlan_unclassified = build_garlan_unclassified_ocean_mask(ocean).astype(np.uint8)
    training_set = set(training_feature_names())
    gap_summary = {}
    gap_csv = OUT_MASKS / "ocean_gap_summary.csv"
    if gap_csv.exists():
        for _, r in pd.read_csv(gap_csv).iterrows():
            gap_summary[r["feature"]] = r["gap_source"]
    print(
        "Garlan unclassified ocean cells:",
        int(garlan_unclassified.sum()),
        f"({100 * garlan_unclassified.sum() / ocean.sum():.1f}% of ocean)",
    )

    masks: list[np.ndarray] = []
    rows: list[dict] = []
    base_gaps: dict[str, np.ndarray] = {}

    for idx, fname in enumerate(feature_names):
        arr_final = load_grid(FINAL / fname)
        if arr_final.shape != elev.shape:
            raise ValueError(f"Shape mismatch for {fname}: {arr_final.shape} vs {elev.shape}")

        if is_rivermouth_feature(fname):
            mask = np.zeros_like(elev, dtype=np.uint8)
            source = "rivermouth_offmouth_zero"
        else:
            gap, source = detect_ocean_gap(
                fname,
                arr_final,
                ocean,
                lats,
                lons,
                garlan_unclassified.astype(bool),
                base_gaps,
            )
            if gap.any():
                mask = gap.astype(np.uint8)
            elif (
                fname in training_set
                and (saved_gap := gap_for_feature_name(fname)) is not None
                and saved_gap.any()
            ):
                # Harmonize filled cells with no dedicated detector (e.g. NN imputation).
                mask = saved_gap.astype(np.uint8)
                source = gap_summary.get(fname, "harmonized_ocean_gap")
            else:
                mrow = meta_for(fname, meta_map)
                if mrow:
                    mask, source = metadata_mask(
                        fname, mrow["extrapolation_type"], elev, ocean, shelf, arr_final
                    )
                else:
                    mask = np.zeros_like(elev, dtype=np.uint8)
                    source = "none_observed"

        if fname in training_set:
            base_gaps[fname] = mask.astype(bool)

        mrow = meta_for(fname, meta_map)
        if mrow and mrow["extrapolation_type"] == "Diesing_shelf_nearest_deep_water":
            shelf_mask = shelf.astype(np.uint8)
            if not np.array_equal(mask, shelf_mask):
                combined = (mask.astype(bool) | shelf).astype(np.uint8)
                if combined.sum() > mask.sum():
                    mask = combined
                    if source != "metadata_diesing_shelf":
                        source = f"{source}+diesing_shelf"

        masks.append(mask)
        rows.append(
            {
                "feature_index": idx,
                "feature": fname,
                "mask_source": source,
                "ocean_interpolated": int((mask & ocean).sum()),
                "ocean_interpolated_pct": round(100 * (mask & ocean).sum() / ocean.sum(), 2),
                "shelf_interpolated": int((mask & shelf).sum()),
                "deep_interpolated": int((mask & deep).sum()),
                "empirical_proven": fname in EMPIRICAL_FEATURES,
            }
        )

        if args.plot_empirical_only and (
            fname in EMPIRICAL_FEATURES or source == "garlan_unclassified"
        ):
            slug = fname.replace(".nc", "").replace(".grd", "")
            plot_mask(
                mask,
                lats,
                lons,
                f"Interpolation mask (1=filled): {fname}",
                OUT_PLOTS / f"interpolation_mask_{slug}.png",
                ocean,
                shelf,
                deep,
            )

    stack = np.stack(masks, axis=-1)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)
    np.save(OUT_MASKS / "interpolation_mask_stack.npy", stack)
    (OUT_MASKS / "interpolation_mask_feature_names.txt").write_text(
        "\n".join(feature_names) + "\n"
    )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_MASKS / "interpolation_mask_summary.csv", index=False)

    print(f"Saved mask stack {stack.shape} -> {OUT_MASKS / 'interpolation_mask_stack.npy'}")
    print(f"Saved summary -> {OUT_MASKS / 'interpolation_mask_summary.csv'}")
    print("\nEmpirical masks:")
    print(
        summary[summary["empirical_proven"]][
            ["feature", "mask_source", "ocean_interpolated", "shelf_interpolated", "deep_interpolated"]
        ].to_string(index=False)
    )
    print("\nMask source counts:")
    print(summary["mask_source"].value_counts().to_string())


if __name__ == "__main__":
    main()
