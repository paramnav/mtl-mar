"""Display paper Table-2 metrics / SI scatters / MAR maps inside train notebooks.

Paper maps are not all in the same geographic orientation:
  - MTL, MTL+attention, TabPFN: already north-up and land-zeroed (match island_map)
  - RF / XGB / CatBoost / KNN / FFNN / FFNN+attention: south-up → flipud before land mask

Plot style matches ``nn4sedratetoc/scripts/replot_paper_maps_robinson.py``:
  cmcrameri batlow, MAR in g cm^-2 yr^-1, log scale 1e-4 … 1e1, Robinson projection.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

# Models whose stored .npy grids are south-up relative to island_map (north-up).
SOUTH_UP_MODELS = frozenset(
    {
        "random_forest",
        "xgboost",
        "catboost",
        "knn",
        "ffnn",
        "ffnn_attention",
    }
)

SI_SCATTER_STEM = {
    "random_forest": "scatter_entire_rf",
    "xgboost": "scatter_entire_xgb",
    "catboost": "scatter_entire_catb",
    "knn": "scatter_entire_knn",
    "tabpfn": "scatter_entire_tabpfn",
    "ffnn": "scatter_entire_nn",
    "mtl": "scatter_entire_mtl",
    "ffnn_attention": "scatter_entire_nn_attention",
    "mtl_attention": "scatter_entire_mtl_attention",
}

MAP_BASENAME = {
    "random_forest": "prediction_map_MAR_entire_rf_n700.npy",
    "xgboost": "prediction_map_MAR_entire_xgb_n1000.npy",
    "catboost": "prediction_map_MAR_entire_catb_n1000.npy",
    "knn": "prediction_map_MAR_entire_knn_k6.npy",
    "tabpfn": "prediction_map_MAR_entire_tabpfn_nomask.npy",
    "ffnn": "prediction_map_SedRate_entire_supervised_nomask_[128, 128, 128].npy",
    "mtl": "prediction_map_MAR_entire_MTL_n_nomask_47.npy",
    "ffnn_attention": "prediction_map_SedRate_entire_supervised_attention_attn_d256_h8_l3_ff2_do0.2.npy",
    "mtl_attention": "prediction_map_MAR_entire_MTL_n_attention_attn_d128_h8_l2_ff2_do0.2.npy",
}

DISPLAY_NAME = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "knn": "KNN",
    "tabpfn": "TabPFN",
    "ffnn": "FFNN",
    "mtl": "MTL",
    "ffnn_attention": "FFNN + attention",
    "mtl_attention": "MTL + attention",
}


def resolve_project_root(start: Path | None = None) -> Path:
    root = (start or Path().resolve()).resolve()
    if root.name == "notebooks_2":
        return root.parent
    if (root / "notebooks_2").is_dir() and (root / "data").is_dir():
        return root
    for p in [root, *root.parents]:
        if (p / "notebooks_2").is_dir() and (p / "data").is_dir():
            return p
    return root


def paper_out_dir(project_root: Path) -> Path:
    return project_root / "data/output/predictionmaps_SedRate/notebooks_2_paper_table2"


def load_land(project_root: Path) -> np.ndarray:
    land = np.load(project_root / "data/raw/island_map.npy")
    return np.where(np.isnan(land), 1, land).astype(bool)


def align_map_north_up(arr: np.ndarray, model_key: str) -> np.ndarray:
    """Return a north-up copy of a paper prediction map."""
    out = np.asarray(arr, dtype=float).copy()
    if model_key in SOUTH_UP_MODELS:
        out = np.flipud(out)
    return out


def mar_ka_to_plot_yr(mar_ka: np.ndarray, land: np.ndarray, log_floor: float = 1e-4) -> np.ndarray:
    """g cm^-2 ka^-1 → g cm^-2 yr^-1 for paper-style log maps; land → NaN."""
    out = np.asarray(mar_ka, dtype=float) / 1e3
    out = np.clip(out, 0.0, None)
    out[out < log_floor] = log_floor
    out = out.copy()
    out[land] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _ensure_robinson_on_path(project_root: Path) -> None:
    nn4_scripts = project_root.parent / "nn4sedratetoc" / "scripts"
    if nn4_scripts.is_dir() and str(nn4_scripts) not in sys.path:
        sys.path.insert(0, str(nn4_scripts))


def plot_mar_robinson_png(
    mar_ka_north_up: np.ndarray,
    land: np.ndarray,
    *,
    title: str,
    project_root: Path,
    dpi: int = 140,
) -> bytes:
    """Render paper-style Robinson MAR map to PNG bytes."""
    import cmcrameri.cm as cmc

    _ensure_robinson_on_path(project_root)
    from plot_robinson import DATA_CRS, LAND_COLOR, OCEAN_COLOR, PROJ, COAST_COLOR  # noqa: WPS433
    import cartopy.crs as ccrs  # noqa: F401
    from matplotlib.colors import LogNorm

    plot = mar_ka_to_plot_yr(mar_ka_north_up, land)
    plot = np.ma.masked_invalid(plot)
    cmap = cmc.batlow.copy()
    cmap.set_bad(LAND_COLOR)
    norm = LogNorm(vmin=1e-4, vmax=1e1)

    fig = plt.figure(figsize=(12.5, 6.2), dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    ax.set_global()
    ax.set_facecolor(OCEAN_COLOR)
    im = ax.imshow(
        plot,
        origin="upper",
        extent=[-180, 180, -90, 90],
        transform=DATA_CRS,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        zorder=1,
    )
    ax.coastlines(resolution="110m", linewidth=0.35, color=COAST_COLOR, zorder=3)
    gl = ax.gridlines(draw_labels=False, linewidth=0.25, color="#888888", alpha=0.45, linestyle=":")
    gl.xlocator = plt.FixedLocator([-120, -60, 0, 60, 120])
    gl.ylocator = plt.FixedLocator([-60, -30, 0, 30, 60])
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.72, pad=0.02, aspect=28)
    cbar.set_label(r"Mass accumulation rate ($g/cm^2/yr$)", fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ax.set_title(title, fontsize=12, pad=8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def show_paper_models(model_keys: list[str] | str, project_root: Path | None = None) -> pd.DataFrame:
    """Show Table-2 row(s), SI scatter(s), and paper map(s) for one or more models."""
    if isinstance(model_keys, str):
        model_keys = [model_keys]

    root = resolve_project_root(project_root)
    out = paper_out_dir(root)
    paper_csv = out / "table2_paper_metrics.csv"
    si_dir = out / "scatters" / "si_paper"
    maps_dir = out / "maps_paper_names"

    if not paper_csv.exists():
        raise FileNotFoundError(
            f"Missing {paper_csv}\n"
            "Run: sbatch notebooks_2/batchfiles/batchfile_reproduce_paper_table2"
        )

    paper = pd.read_csv(paper_csv)
    land = load_land(root)
    rows = []

    for key in model_keys:
        if key not in MAP_BASENAME:
            raise KeyError(f"Unknown model key {key!r}; expected one of {sorted(MAP_BASENAME)}")

        sub = paper.loc[paper["model"] == key]
        if sub.empty:
            raise KeyError(f"{key} not in {paper_csv}")
        row = sub.iloc[0]
        rows.append(row)

        display(Markdown(f"### {DISPLAY_NAME[key]} — Table 2"))
        display(
            pd.DataFrame(
                [
                    {
                        "model": DISPLAY_NAME[key],
                        "train_MAE": row["train_mae"],
                        "train_CC": row["train_cc"],
                        "test_MAE": row["test_mae"],
                        "test_CC": row["test_cc"],
                        "budget_Pg_yr": row["budget_pg_yr"],
                    }
                ]
            )
        )

        stem = SI_SCATTER_STEM[key]
        scatter = si_dir / f"{stem}.png"
        display(Markdown(f"#### SI scatter (`{stem}`)"))
        if scatter.exists():
            display(Image(filename=str(scatter), width=420))
        else:
            print("MISSING scatter:", scatter)

        map_path = maps_dir / MAP_BASENAME[key]
        display(Markdown(f"#### Prediction map (paper orientation + batlow)"))
        if not map_path.exists():
            print("MISSING map:", map_path)
            continue
        mar_ka = align_map_north_up(np.load(map_path), key)
        orient = "flipud (was south-up)" if key in SOUTH_UP_MODELS else "as stored (north-up)"
        print(f"{key}: orientation fix = {orient}; map = {map_path.name}")
        png = plot_mar_robinson_png(
            mar_ka,
            land,
            title=f"{DISPLAY_NAME[key]} — Table 2 MAR",
            project_root=root,
        )
        display(Image(data=png))

    return pd.DataFrame(rows)
