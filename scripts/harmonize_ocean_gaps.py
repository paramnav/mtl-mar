#!/usr/bin/env python3
"""Harmonize ocean gaps: NN cross-feature imputation + per-feature gap masks.

Gap cells are filled with a per-feature tuned neural network
(``nn_feature_imputation``), using only natively observed values of other
features as predictors (imputed values are never used as predictors).  The
imputer runs a random hyperparameter search per feature scored by validation
R^2 on held-out native cells; metrics are written to ml_imputation_summary.csv.
Rivermouth layers keep zero-fill; r50km companions are rebuilt from imputed
base layers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from feature_update_core import FEATURE_STACK, build_r50km_companion, write_stack_grid
from mask_gap_core import (
    OCEAN_GAP_NAMES,
    OCEAN_GAP_STACK,
    OUT_MASKS,
    build_garlan_unclassified_ocean_mask,
    detect_ocean_gap,
    is_garlan_feature,
    is_rivermouth_feature,
    garlan_gap_mask,
    load_grid_with_coords,
    resolve_r50_base,
    training_feature_names,
)
from nn_feature_imputation import DEFAULT_HP_TRIALS, impute_stack_nn
from spatial_interpolation import DEFAULT_POLAR_LAT_THRESHOLD_DEG, fill_ocean_gaps, smooth_circular_ocean_gaps

WAVE_DIRECTION_BASE = "SS_WAVE_DIRECTION_DEG_2018_2023_WAVEWATCH3x.5m.grd"

# Ordered list of the four compositional sediment-type probability features.
# After independent ML imputation their sum may deviate from 1; renormalization
# restores the simplex constraint at gap cells.
PROBABILITY_FEATURES = [
    "probabilities_Calc.Sed_mod.nc",
    "probabilities_Clay_mod.nc",
    "probabilities_Dia.Ooze_mod.nc",
    "probabilities_Lith.Sed_mod.nc",
]


def harmonize_grid_zero(fname: str, gap: np.ndarray, ocean: np.ndarray) -> int:
    path = FEATURE_STACK / fname
    ds = xr.open_dataset(path)
    template = ds["data"]
    arr = np.asarray(template.values, dtype=np.float32)
    lats = np.asarray(ds["lat"].values)
    lons = np.asarray(ds["lon"].values)
    ds.close()
    filled = fill_ocean_gaps(arr, gap, ocean, method="zero")
    return write_stack_grid(filled, path, template=template, lats=lats, lons=lons)


def write_filled_grid(fname: str, filled: np.ndarray) -> int:
    path = FEATURE_STACK / fname
    ds = xr.open_dataset(path)
    template = ds["data"]
    lats = np.asarray(ds["lat"].values)
    lons = np.asarray(ds["lon"].values)
    ds.close()
    return write_stack_grid(filled, path, template=template, lats=lats, lons=lons)


def smooth_wave_direction_boundaries(
    gap_by_name: dict[str, np.ndarray],
    ocean: np.ndarray,
) -> None:
    """Soften native/imputed seams on the base wave-direction grid."""
    if WAVE_DIRECTION_BASE not in gap_by_name:
        return
    gap = gap_by_name[WAVE_DIRECTION_BASE]
    if not (gap & ocean).any():
        return
    arr, _, _ = load_grid_with_coords(FEATURE_STACK / WAVE_DIRECTION_BASE)
    smoothed = smooth_circular_ocean_gaps(arr, gap, ocean)
    n = write_filled_grid(WAVE_DIRECTION_BASE, smoothed)
    print(f"  {WAVE_DIRECTION_BASE}: boundary-smoothed, finite={n:,}")


def detect_all_gaps(
    names: list[str],
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    garlan: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    name_set = set(names)
    gap_by_name: dict[str, np.ndarray] = {}
    rows: list[dict] = []

    for fname in names:
        if resolve_r50_base(fname, name_set) is not None:
            continue
        arr, _, _ = load_grid_with_coords(FEATURE_STACK / fname)
        gap, source = detect_ocean_gap(fname, arr, ocean, lats, lons, garlan, gap_by_name)
        gap_by_name[fname] = gap
        rows.append(
            {
                "feature": fname,
                "gap_source": source,
                "ocean_gap_cells": int(gap.sum()),
                "ocean_gap_pct": round(100 * gap.sum() / ocean.sum(), 2),
            }
        )
        print(f"  {fname}: {int(gap.sum()):,} gap cells ({source})")

    for fname in names:
        if fname in gap_by_name:
            continue
        arr, _, _ = load_grid_with_coords(FEATURE_STACK / fname)
        gap, source = detect_ocean_gap(fname, arr, ocean, lats, lons, garlan, gap_by_name)
        gap_by_name[fname] = gap
        rows.append(
            {
                "feature": fname,
                "gap_source": source,
                "ocean_gap_cells": int(gap.sum()),
                "ocean_gap_pct": round(100 * gap.sum() / ocean.sum(), 2),
            }
        )
        print(f"  {fname}: {int(gap.sum()):,} gap cells ({source})")
    return gap_by_name, rows


def load_gap_dict(all_names: list[str]) -> dict[str, np.ndarray]:
    stack = np.load(OCEAN_GAP_STACK)
    return {all_names[i]: stack[:, :, i].astype(bool) for i in range(len(all_names))}


def save_gap_stack(all_names: list[str], gap_by_name: dict[str, np.ndarray], rows: list[dict]) -> None:
    ordered = np.stack([gap_by_name[f].astype(np.uint8) for f in all_names], axis=-1)
    OUT_MASKS.mkdir(parents=True, exist_ok=True)
    np.save(OCEAN_GAP_STACK, ordered)
    OCEAN_GAP_NAMES.write_text("\n".join(all_names) + "\n")
    pd.DataFrame(rows).to_csv(OUT_MASKS / "ocean_gap_summary.csv", index=False)
    print(f"Saved gap stack {ordered.shape} -> {OCEAN_GAP_STACK}")


def rebuild_r50km(
    all_names: list[str],
    gap_by_name: dict[str, np.ndarray],
    *,
    only_bases: list[str] | None = None,
) -> None:
    name_set = set(all_names)
    only_base_set = set(only_bases) if only_bases else None
    print("\nRebuilding r50km companions...")
    for fname in all_names:
        base = resolve_r50_base(fname, name_set)
        if base is None:
            continue
        if only_base_set is not None and base not in only_base_set:
            continue
        gap = gap_by_name[base]
        circular = "WAVE_DIRECTION" in fname
        print(f"  {base} -> {fname}")
        # The base layer is already gap-filled (NN/ML imputed, or zero-filled for
        # rivermouth/Garlan), so the neighborhood mean is taken over the complete
        # imputed field. Gap cells therefore receive a real regional average
        # instead of a 0 sentinel (which z-scored to extreme outliers, e.g. the
        # -513 sigma seawater-density spikes that produced the coastal ring).
        build_r50km_companion(
            base,
            fname,
            circular=circular,
            invalid_mask=None,
            zero_invalid=False,
        )
        gap_by_name[fname] = gap.copy()


def _rows_from_gap_dict(
    gap_by_name: dict[str, np.ndarray],
    all_names: list[str],
    ocean: np.ndarray,
    existing_csv: Path | None = None,
) -> list[dict]:
    """Build gap summary rows from *gap_by_name* so the CSV stays in sync with the npy stack.

    Cell counts always come from *gap_by_name* (authoritative); ``gap_source`` labels are
    preserved from *existing_csv* when available so the audit history is not lost.
    """
    source_map: dict[str, str] = {}
    if existing_csv is not None and existing_csv.exists():
        for r in pd.read_csv(existing_csv).to_dict("records"):
            source_map[r["feature"]] = str(r.get("gap_source", "unknown"))
    rows: list[dict] = []
    for fname in all_names:
        if fname not in gap_by_name:
            continue
        gap = gap_by_name[fname]
        rows.append(
            {
                "feature": fname,
                "gap_source": source_map.get(fname, "unknown"),
                "ocean_gap_cells": int(gap.sum()),
                "ocean_gap_pct": round(100.0 * float(gap.sum()) / float(ocean.sum()), 2),
            }
        )
    return rows


def _renormalize_probabilities(
    gap_by_name: dict[str, np.ndarray],
    ocean: np.ndarray,
) -> None:
    """Re-scale ML-imputed probability features so they sum to 1 at gap cells.

    Independent ML imputation produces four probabilities whose sum may deviate
    from the simplex constraint.  Only gap-ocean cells are modified; native cells
    are untouched.  Cells where the sum is effectively zero receive equal weights.
    """
    present = [f for f in PROBABILITY_FEATURES if f in gap_by_name]
    if len(present) < 2:
        return

    # Union of imputed ocean cells across all present probability features
    any_gap = np.zeros(ocean.shape, dtype=bool)
    for fname in present:
        any_gap |= gap_by_name[fname].astype(bool) & ocean
    if not any_gap.any():
        return

    # Load current (ML-filled) grids; clamp to [0, 1] first
    arrays: dict[str, np.ndarray] = {}
    for fname in present:
        arr, _, _ = load_grid_with_coords(FEATURE_STACK / fname)
        arrays[fname] = np.clip(np.asarray(arr, dtype=np.float64), 0.0, 1.0)

    prob_sum: np.ndarray = sum(arrays[f] for f in present)  # type: ignore[assignment]
    deviation_before = float(np.abs(prob_sum[any_gap] - 1.0).mean())

    normalize_mask = any_gap & (prob_sum > 1e-8)
    zero_sum_mask = any_gap & (prob_sum <= 1e-8)

    if normalize_mask.any():
        denom = prob_sum[normalize_mask]
        for fname in present:
            arrays[fname][normalize_mask] /= denom

    if zero_sum_mask.any():
        eq = 1.0 / len(present)
        for fname in present:
            arrays[fname][zero_sum_mask] = eq

    # Write back
    for fname in present:
        n = write_filled_grid(fname, arrays[fname].astype(np.float32))
        print(f"  {fname}: simplex-renormalized, finite={n:,}")

    prob_sum_after: np.ndarray = sum(arrays[f] for f in present)  # type: ignore[assignment]
    deviation_after = float(np.abs(prob_sum_after[any_gap] - 1.0).mean())
    n_cells = int(normalize_mask.sum()) + int(zero_sum_mask.sum())
    print(
        f"  Mean simplex |sum−1|: before={deviation_before:.4f} → after={deviation_after:.2e} "
        f"({n_cells:,} gap cells renormalized)"
    )


def harmonize_ml(
    all_names: list[str],
    gap_by_name: dict[str, np.ndarray],
    ocean: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    only: list[str] | None,
    max_train_samples: int,
    min_native_predictors: int,
    hp_trials: int = DEFAULT_HP_TRIALS,
    polar_lat_threshold: float = DEFAULT_POLAR_LAT_THRESHOLD_DEG,
) -> None:
    """NN-impute base gap features; zero-fill rivermouth and Garlan layers."""
    name_set = set(all_names)
    write_names = [n for n in all_names if only is None or n in set(only)]

    ml_targets = {
        f
        for f in write_names
        if resolve_r50_base(f, name_set) is None
        and not is_rivermouth_feature(f)
        and not is_garlan_feature(f)
        and (gap_by_name[f] & ocean).any()
    }
    if ml_targets:
        print(f"\nNN imputation for {len(ml_targets)} base feature(s)...")
        filled, stats_rows = impute_stack_nn(
            all_names,
            gap_by_name,
            ocean,
            lats,
            lons,
            only=list(ml_targets),
            max_train_samples=max_train_samples,
            min_native_predictors=min_native_predictors,
            hp_trials=hp_trials,
            polar_lat_threshold=polar_lat_threshold,
        )
        for fname, grid in filled.items():
            n = write_filled_grid(fname, grid)
            row = next(r for r in stats_rows if r["feature"] == fname)
            val_info = ""
            if "val_r2" in row:
                val_info = f"  val_R2={row['val_r2']:.4f}  val_RMSE={row['val_rmse']:.4g}"
            print(
                f"  {fname}: finite={n:,}  "
                f"ml={row['ml_filled']:,}  spatial_fallback={row['spatial_fallback']:,}  "
                f"polar_idw={row.get('polar_idw_filled', 0):,}{val_info}"
            )
        stats_path = OUT_MASKS / "ml_imputation_summary.csv"
        pd.DataFrame(stats_rows).to_csv(stats_path, index=False)
        print(f"Saved NN imputation stats -> {stats_path}")

    # Enforce simplex constraint: probabilities_* must sum to 1 at gap cells.
    # Independent ML models may violate this; renormalize without touching native cells.
    prob_targets = [f for f in write_names if f in set(PROBABILITY_FEATURES)]
    if prob_targets:
        print("\nRenormalizing probability features to unit sum at gap cells...")
        _renormalize_probabilities(gap_by_name, ocean)

    garlan = [
        f
        for f in write_names
        if is_garlan_feature(f)
    ]
    if garlan:
        garlan_gap = garlan_gap_mask(ocean)
        for fname in garlan:
            gap_by_name[fname] = garlan_gap.copy()
        # Litho type maps: NN lithology-class prediction for unclassified ocean
        # (replaces legacy zero-fill). The six one-hot type grids are rewritten
        # jointly so exactly one class is active per filled cell. Grain-size
        # derivatives (non type-map Garlan layers) keep zero-fill.
        litho_types = [f for f in garlan if "litho_maps_type" in f]
        other_garlan = [f for f in garlan if "litho_maps_type" not in f]
        if litho_types and garlan_gap.any():
            print(
                f"\nNN lithology-class fill for {len(litho_types)} litho type map(s) "
                "(unclassified ocean predicted from environmental features)..."
            )
            from apply_litho_nn_fill import run_litho_nn_fill

            val_acc = run_litho_nn_fill()
            print(f"  litho NN fill done (val accuracy={val_acc:.4f})")
        if other_garlan and garlan_gap.any():
            print(
                f"\nZero-fill for {len(other_garlan)} Garlan grain-size layer(s)..."
            )
            for fname in other_garlan:
                n = harmonize_grid_zero(fname, garlan_gap, ocean)
                print(f"  {fname}: finite cells={n:,}")

    rivermouth = [f for f in write_names if is_rivermouth_feature(f) and (gap_by_name[f] & ocean).any()]
    if rivermouth:
        print(f"\nZero-fill for {len(rivermouth)} rivermouth feature(s)...")
        for fname in rivermouth:
            n = harmonize_grid_zero(fname, gap_by_name[fname], ocean)
            print(f"  {fname}: finite cells={n:,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-r50km", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only", nargs="+", metavar="FILE")
    parser.add_argument("--r50km-only", action="store_true")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=400_000,
        help="Max training cells per feature (default: 400000)",
    )
    parser.add_argument(
        "--min-native-predictors",
        type=int,
        default=5,
        help="Spatial fallback when fewer native predictors at a gap cell",
    )
    parser.add_argument(
        "--hp-trials",
        type=int,
        default=DEFAULT_HP_TRIALS,
        help=f"NN hyperparameter search trials per feature (default: {DEFAULT_HP_TRIALS})",
    )
    parser.add_argument(
        "--polar-lat-threshold",
        type=float,
        default=DEFAULT_POLAR_LAT_THRESHOLD_DEG,
        help=(
            "Use Laplacian IDW instead of ML/NN for ocean gap cells with "
            f"|latitude| >= this value (default: {DEFAULT_POLAR_LAT_THRESHOLD_DEG})"
        ),
    )
    parser.add_argument(
        "--smooth-wave-direction-only",
        action="store_true",
        help="Only relax imputed wave-direction seams (no ML re-imputation)",
    )
    args = parser.parse_args()

    all_names = training_feature_names()
    detect_names = all_names
    if args.only:
        detect_names = [n for n in all_names if n in set(args.only)]
        if not detect_names:
            raise SystemExit(f"No matches for --only: {args.only}")

    elev, lats, lons = load_grid_with_coords(FEATURE_STACK / "GL_ELEVATION_M_ASL_ETOPO2022.5m.grd")
    ocean = elev < 0
    garlan = build_garlan_unclassified_ocean_mask(ocean)

    if args.r50km_only:
        gap_by_name = load_gap_dict(all_names)
    elif args.smooth_wave_direction_only:
        if OCEAN_GAP_STACK.exists():
            gap_by_name = load_gap_dict(all_names)
        else:
            raise SystemExit("ocean_gap_stack.npy missing; run gap detection first")
        print("\nSmoothing wave direction boundaries...")
        smooth_wave_direction_boundaries(gap_by_name, ocean)
    else:
        print(f"Detecting ocean gaps for {len(detect_names)} feature(s)...")
        partial, rows = detect_all_gaps(detect_names, ocean, lats, lons, garlan)
        if OCEAN_GAP_STACK.exists():
            gap_by_name = load_gap_dict(all_names)
            if args.only:
                # Update only the requested features; merge rows from fresh detection
                # with the existing CSV rows for all other features.
                gap_by_name.update(partial)
                row_map = {}
                if (OUT_MASKS / "ocean_gap_summary.csv").exists():
                    row_map = {
                        r["feature"]: r
                        for r in pd.read_csv(OUT_MASKS / "ocean_gap_summary.csv").to_dict("records")
                    }
                for r in rows:
                    row_map[r["feature"]] = r
                rows = list(row_map.values())
            else:
                # Full re-run: gap_by_name comes from the saved stack (may reflect a
                # prior imputation run where gap cells have been filled).  Re-derive rows
                # from gap_by_name so the CSV cell-counts always match the npy stack.
                # Source labels are preserved from the existing CSV for audit continuity.
                rows = _rows_from_gap_dict(
                    gap_by_name, all_names, ocean,
                    existing_csv=OUT_MASKS / "ocean_gap_summary.csv",
                )
        else:
            gap_by_name = partial

        save_gap_stack(all_names, gap_by_name, rows)

        if args.dry_run:
            return

        harmonize_ml(
            all_names,
            gap_by_name,
            ocean,
            lats,
            lons,
            only=detect_names if args.only else None,
            max_train_samples=args.max_train_samples,
            min_native_predictors=args.min_native_predictors,
            hp_trials=args.hp_trials,
            polar_lat_threshold=args.polar_lat_threshold,
        )

    if args.skip_r50km:
        return

    only_bases = None
    if args.only:
        name_set = set(all_names)
        only_bases = []
        for n in detect_names:
            base = resolve_r50_base(n, name_set)
            only_bases.append(base if base is not None else n)
        # Also rebuild r50 companions when the base itself was requested
        only_bases = list(dict.fromkeys(only_bases))

    rebuild_r50km(all_names, gap_by_name, only_bases=only_bases)
    save_gap_stack(all_names, gap_by_name, pd.read_csv(OUT_MASKS / "ocean_gap_summary.csv").to_dict("records"))
    print("Done.")


if __name__ == "__main__":
    main()
