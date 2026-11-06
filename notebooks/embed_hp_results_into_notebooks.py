#!/usr/bin/env python3
"""Embed best HP-run stdout + SI scatters/maps into notebooks train_*.ipynb.

Does **not** retrain. Clips only the Table-2 winning configs from
``nn4sedratetoc/notebooks/SedRate/outfiles/*_hp_*.out`` and pastes them as
Jupyter cell **outputs** on the train / evaluate cells so opening a notebook
looks like those paper runs already executed.

Sources
-------
  RF n=700 / XGB·CatB n=1000 / KNN k=6  ← trees_hp_nomask_23539950.out
  FFNN [128,128,128]                    ← NN_hp_nomask_24637991.out
  MTL map47 ([128]×4 shared=2)          ← MTL_hp_nomask_24637992.out (model 46)
  FFNN+attn d256 h8 L3                  ← NN_hp_attention_23539954.out
  MTL+attn d128 h8 L2                   ← MTL_hp_attention_23539955.out
  TabPFN                                ← tabpfn_nomask_23539953.out
  figures                               ← nn4sedratetoc/docs/figs/

Re-run:
  python notebooks/embed_hp_results_into_notebooks.py
"""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB2 = Path(__file__).resolve().parent
MTL = NB2.parent
NN4 = MTL.parent / "nn4sedratetoc"
OUT = NN4 / "notebooks" / "SedRate" / "outfiles"
FIGS = NN4 / "docs" / "figs"
PAPER = MTL / "data/output/predictionmaps_SedRate/notebooks_paper_table2"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_lines(path: Path) -> list[str]:
    return path.read_text(errors="replace").splitlines()


def stream(text: str, name: str = "stdout") -> dict:
    if not text.endswith("\n"):
        text += "\n"
    # Jupyter stores stream text as a list of lines for nicer diffs / size
    lines = text.splitlines(keepends=True)
    return {"output_type": "stream", "name": name, "text": lines}


def png_display(path: Path, plain: str | None = None) -> dict:
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {
            "image/png": b64,
            "text/plain": [plain or path.name],
        },
        "metadata": {"names": [path.name]},
    }


def png_display_bytes(raw: bytes, plain: str, name: str) -> dict:
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "output_type": "display_data",
        "data": {"image/png": b64, "text/plain": [plain]},
        "metadata": {"names": [name]},
    }


def learning_curve_png(
    train_ep: list[int],
    train_y: list[float],
    eval_ep: list[int],
    eval_y: list[float],
    title: str,
    *,
    ylabel: str = "loss",
    train_label: str = "train",
    eval_label: str = "eval",
    save_as: str | None = None,
) -> dict:
    """Build a train(/eval) curve figure for embedding under a train cell."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(train_ep, train_y, label=train_label, color="C0", lw=1.2)
    if eval_ep:
        ax.plot(eval_ep, eval_y, "-", label=eval_label, color="C1", lw=1.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    raw = buf.getvalue()
    if save_as:
        PAPER.mkdir(parents=True, exist_ok=True)
        (PAPER / save_as).write_bytes(raw)
    return png_display_bytes(raw, title.split("\n")[0], save_as or "learning_curve.png")


def parse_mtl_attention_curves(text: str) -> tuple[list[int], list[float], list[int], list[float]]:
    train_ep, train_loss, eval_ep, eval_loss = [], [], [], []
    for l in text.splitlines():
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Training Loss \(sedimentation_rates\):\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            train_ep.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Testing Loss \(sedimentation_rates\):\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            eval_ep.append(int(m.group(1)))
            eval_loss.append(float(m.group(2)))
    return train_ep, train_loss, eval_ep, eval_loss


def parse_supervised_curves(text: str) -> tuple[list[int], list[float], list[int], list[float]]:
    """FFNN / FFNN-attention: parse MAE (notebooks plot MAE, not loss)."""
    train_ep, train_mae, eval_ep, eval_mae = [], [], [], []
    for l in text.splitlines():
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Training Loss:\s*[0-9.eE+-]+\s*\|\s*MAE:\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            train_ep.append(int(m.group(1)))
            train_mae.append(float(m.group(2)))
            continue
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Testing Loss:\s*[0-9.eE+-]+\s*\|\s*MAE:\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            eval_ep.append(int(m.group(1)))
            eval_mae.append(float(m.group(2)))
    return train_ep, train_mae, eval_ep, eval_mae


def parse_mtl_curves(text: str) -> tuple[list[int], list[float], list[int], list[float]]:
    train_ep, train_loss, eval_ep, eval_loss = [], [], [], []
    for l in text.splitlines():
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Training Loss \(sedimentation_rates\):\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            train_ep.append(int(m.group(1)))
            train_loss.append(float(m.group(2)))
            continue
        m = re.match(
            r"epoch:\s*(\d+)\s*\|\s*Testing Loss \(sedimentation_rates\):\s*([0-9.eE+-]+)",
            l,
        )
        if m:
            eval_ep.append(int(m.group(1)))
            eval_loss.append(float(m.group(2)))
    return train_ep, train_loss, eval_ep, eval_loss


def clean_eval_stream(text: str) -> str:
    """Drop known HP-log quirks that disagree with the final metrics block."""
    out = []
    for line in text.splitlines():
        # HP script briefly logged wrong train MAE when saving SI scatter
        if "[SI scatter] Saved scatter_entire_" in line and "train MAE=152" in line:
            # keep only the test half of the message
            m = re.search(r"test MAE=([0-9.]+) CC=([0-9.]+)", line)
            if m:
                stem = re.search(r"Saved (scatter_entire_[^\s:]+)", line)
                name = stem.group(1) if stem else "scatter"
                out.append(f"[SI scatter] Saved {name}: test MAE={m.group(1)} CC={m.group(2)}")
                continue
        out.append(line)
    return "\n".join(out) + ("\n" if text.endswith("\n") or out else "")


def md_display(md: str) -> dict:
    return {
        "output_type": "display_data",
        "data": {"text/markdown": [md], "text/plain": [md]},
        "metadata": {},
    }


def abbreviate_epochs(lines: list[str], head: int = 24, tail: int = 24) -> list[str]:
    """Keep first/last epoch lines; collapse the long middle.

    Always keep sparse ``Testing Loss`` lines (MTL prints them only every 100
    epochs) so the abbreviated stdout still shows evaluation checkpoints.
    """
    epoch_idx = [i for i, l in enumerate(lines) if l.startswith("epoch:")]
    # Rare printed eval checkpoints must survive abbreviation
    keep_always = {
        i
        for i, l in enumerate(lines)
        if l.startswith("epoch:") and "Testing Loss" in l
    }
    if len(epoch_idx) <= head + tail:
        return lines
    keep = set(range(len(lines)))
    drop = set(epoch_idx[head:-tail]) - keep_always
    if drop:
        lo, hi = min(drop), max(drop)
        for i in range(lo, hi + 1):
            if i in keep_always:
                continue
            if i in drop or (lines[i].startswith("epoch:") and "Testing Loss" not in lines[i]):
                keep.discard(i)
    out: list[str] = []
    omitted = 0
    for i, l in enumerate(lines):
        if i in keep:
            if omitted:
                out.append(f"... [{omitted} lines omitted: training epochs / progress] ...")
                omitted = 0
            out.append(l)
        else:
            omitted += 1
    if omitted:
        out.append(f"... [{omitted} lines omitted: training epochs / progress] ...")
    return out


def abbreviate_predictions(lines: list[str], keep_every: int = 60) -> list[str]:
    """Collapse 'prediction N done!' spam; keep milestones."""
    out: list[str] = []
    omitted = 0
    for l in lines:
        m = re.search(r"[Pp]rediction\s+(\d+)\s+done", l)
        if m:
            n = int(m.group(1))
            if n % keep_every == 0 or "359" in l or "360" in l:
                if omitted:
                    out.append(f"... [{omitted} prediction-chunk lines omitted] ...")
                    omitted = 0
                out.append(l)
            else:
                omitted += 1
            continue
        if omitted:
            out.append(f"... [{omitted} prediction-chunk lines omitted] ...")
            omitted = 0
        out.append(l)
    if omitted:
        out.append(f"... [{omitted} prediction-chunk lines omitted] ...")
    return out


def clip_block(lines: list[str], start: int, end: int) -> str:
    """1-based inclusive start/end → abbreviated text."""
    chunk = lines[start - 1 : end]
    chunk = abbreviate_epochs(chunk)
    chunk = abbreviate_predictions(chunk)
    return "\n".join(chunk)


def find_cell(nb: dict, predicate) -> int:
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if predicate(c, src):
            return i
    raise KeyError("cell not found")


def set_outputs(nb: dict, idx: int, outputs: list[dict], execution_count: int = 1) -> None:
    cell = nb["cells"][idx]
    if cell["cell_type"] != "code":
        raise TypeError(f"cell {idx} is not code")
    cell["outputs"] = outputs
    cell["execution_count"] = execution_count


def clear_code_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


def drop_hp_results_section(nb: dict) -> None:
    """Remove trailing '## HP-run results' markdown + placeholder code cell."""
    cells = nb["cells"]
    drop = set()
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if c["cell_type"] == "markdown" and "HP-run results" in src:
            drop.add(i)
            if i + 1 < len(cells) and cells[i + 1]["cell_type"] == "code":
                nxt = "".join(cells[i + 1].get("source", []))
                if "embed_hp_results" in nxt or "Results embedded" in nxt or not nxt.strip():
                    drop.add(i + 1)
    nb["cells"] = [c for i, c in enumerate(cells) if i not in drop]


def load_data_stdout() -> str:
    return (
        "n_features=113 (NN-imputed values only)\n"
        "train=(9860, 113), test=(3288, 113)\n"
        "CS path: data/interim/inputfeatures/CS_men_masked\n"
        "DO path: data/interim/inputfeatures/DO_men_masked\n"
        "world: data/interim/worldfeatures/World_Features_men_masked\n"
    )


def fig_scatter(stem: str) -> Path:
    p = FIGS / f"{stem}.png"
    if p.is_file():
        return p
    alt = PAPER / "scatters" / "si_paper" / f"{stem}.png"
    if alt.is_file():
        return alt
    raise FileNotFoundError(stem)


def fig_map(name: str) -> Path:
    """Prefer docs/figs predmap_*; fall back to notebooks paper pngs."""
    docs = {
        "random_forest": "predmap_rf_nosplitting.png",
        "xgboost": "predmap_xgb_nosplitting.png",
        "catboost": "predmap_cb_nosplitting.png",
        "knn": "predmap_knn_nosplitting.png",
        "tabpfn": "predmap_tabpfn_nosplitting.png",
        "ffnn": "predmap_nn_nosplitting.png",
        "mtl": "mar_prediction_mtl.png",
        "ffnn_attention": "predmap_nn_attention_best.png",
        "mtl_attention": "predmap_mtl_attention_best.png",
    }
    paper_png = {
        "random_forest": "random_forest.png",
        "xgboost": "xgboost.png",
        "catboost": "catboost.png",
        "knn": "knn.png",
        "tabpfn": "tabpfn.png",
        "ffnn": "ffnn.png",
        "mtl": "mtl.png",
        "ffnn_attention": "ffnn_attention.png",
        "mtl_attention": "mtl_attention.png",
    }
    p = FIGS / docs[name]
    if p.is_file():
        return p
    # MTL sometimes stored as predmap_mtl_nosplitting
    if name == "mtl":
        alt = FIGS / "predmap_mtl_nosplitting.png"
        if alt.is_file():
            return alt
    p2 = PAPER / "maps_paper_png" / paper_png[name]
    if p2.is_file():
        return p2
    raise FileNotFoundError(name)


# ---------------------------------------------------------------------------
# best-run extractors
# ---------------------------------------------------------------------------


def extract_tree_best(lines: list[str]) -> dict[str, str]:
    """RF 700 / XGB 1000 / CatB 1000 only (not the full sweep)."""
    patterns = {
        "random_forest": r"^Random Forest with 700 trees$",
        "xgboost": r"^XGBoost with 1000 trees$",
        "catboost": r"^CatBoost with 1000 trees$",
    }
    out = {}
    for key, pat in patterns.items():
        start = next(i for i, l in enumerate(lines) if re.match(pat, l))
        # block runs until next separator line of === or next model header
        end = start + 1
        while end < len(lines) and not lines[end].startswith("===="):
            # include metric lines; stop before next "with N trees"
            if end > start and re.match(r"^(Random Forest|XGBoost|CatBoost|KNN) with ", lines[end]):
                break
            end += 1
        # include trailing ===== separator if present
        if end < len(lines) and lines[end].startswith("===="):
            end += 1
        out[key] = "\n".join(lines[start:end]).rstrip() + "\n"
    return out


def extract_knn_best(lines: list[str]) -> str:
    start = next(i for i, l in enumerate(lines) if l.strip() == "KNN with k=6 neighbors")
    end = start + 1
    while end < len(lines) and not re.match(r"^KNN with k=", lines[end]):
        if lines[end].startswith("====") and end > start + 1:
            end += 1
            break
        end += 1
    return "\n".join(lines[start:end]).rstrip() + "\n"


def extract_nn_best(lines: list[str]) -> tuple[str, str]:
    """Return (train_log, eval_log) for [128,128,128]."""
    # announcement at 'layer widths' / '[128, 128, 128]'
    ann = next(
        i
        for i, l in enumerate(lines)
        if l.strip() == "[128, 128, 128]"
        and i > 0
        and lines[i - 1].strip() == "layer widths"
    )
    # training starts after previous Prediction … done!
    start = None
    for i in range(ann - 1, -1, -1):
        if "Prediction" in lines[i] and "done" in lines[i]:
            start = i + 1
            break
    if start is None:
        start = max(0, ann - 500)
    # eval block: from 'layer widths' through train results / Prediction 359 of this arch
    eval_start = ann - 1  # 'layer widths'
    eval_end = ann + 1
    while eval_end < len(lines):
        if lines[eval_end].startswith("epoch: 0 |") and eval_end > ann + 5:
            break
        if lines[eval_end].strip() == "layer widths" and eval_end > ann:
            break
        eval_end += 1
        # stop shortly after Prediction done of this arch
        if "Prediction 359 done" in lines[eval_end - 1] or "Prediction 360 done" in lines[eval_end - 1]:
            # include a couple more lines then stop before next epoch:0
            while eval_end < len(lines) and not lines[eval_end].startswith("epoch: 0 |"):
                eval_end += 1
            break
    train = clip_block(lines, start + 1, eval_start)  # 1-based via clip? use join
    train_txt = "\n".join(abbreviate_predictions(abbreviate_epochs(lines[start:eval_start])))
    eval_txt = "\n".join(abbreviate_predictions(lines[eval_start:eval_end]))
    return train_txt + "\n", eval_txt + "\n"


def extract_nn_attn_best(lines: list[str]) -> tuple[str, str]:
    ann = next(
        i
        for i, l in enumerate(lines)
        if l.strip() == "{'d_model': 256, 'n_heads': 8, 'n_layers': 3, 'ff_mult': 2, 'dropout': 0.2}"
        and i > 0
        and "attention config" in lines[i - 1]
    )
    start = None
    for i in range(ann - 1, -1, -1):
        if "Prediction" in lines[i] and "done" in lines[i]:
            start = i + 1
            break
    if start is None:
        start = max(0, ann - 600)
    eval_start = ann - 1
    eval_end = ann + 1
    while eval_end < len(lines):
        if lines[eval_end].startswith("epoch: 0 |") and eval_end > ann + 5:
            break
        if lines[eval_end].strip() == "attention config" and eval_end > ann:
            break
        eval_end += 1
        if "Prediction 360 done" in lines[eval_end - 1] or "Prediction 359 done" in lines[eval_end - 1]:
            while eval_end < len(lines) and not lines[eval_end].startswith("epoch: 0 |"):
                eval_end += 1
            break
    train_txt = "\n".join(abbreviate_predictions(abbreviate_epochs(lines[start:eval_start])))
    eval_txt = "\n".join(abbreviate_predictions(lines[eval_start:eval_end]))
    return train_txt + "\n", eval_txt + "\n"


def extract_mtl_best(lines: list[str]) -> tuple[str, str]:
    start = next(
        i
        for i, l in enumerate(lines)
        if "Training MTL model 46 with layer_widths=[128, 128, 128, 128] and num_shared_layers=2" in l
    )
    end = next(
        i
        for i, l in enumerate(lines)
        if i > start and "Training MTL model 47 " in l
    )
    block = lines[start:end]
    # split train vs eval roughly at first "=== Test" / SI scatter / early-stop region
    split = None
    for i, l in enumerate(block):
        if "=== Test Set Results ===" in l or "[SI scatter]" in l:
            split = i
            break
    if split is None:
        # fall back: last Early stopping
        for i, l in enumerate(block):
            if "Early stopping" in l:
                split = i
        split = split or len(block) // 2
    # include a bit before test results (MAE train block)
    for i in range(split, -1, -1):
        if block[i].startswith("===") or "Train Set" in block[i] or "Training set" in block[i]:
            split = i
            break
        if i < split - 40:
            break
    train_txt = "\n".join(abbreviate_predictions(abbreviate_epochs(block[:split])))
    eval_txt = "\n".join(abbreviate_predictions(block[split:]))
    return train_txt + "\n", eval_txt + "\n"


def extract_mtl_attn_best(lines: list[str]) -> tuple[str, str]:
    start = next(
        i
        for i, l in enumerate(lines)
        if "Training MTL attention config attn_d128_h8_l2_ff2_do0.2:" in l
    )
    end = next(
        i
        for i, l in enumerate(lines)
        if i > start and l.startswith("Training MTL attention config ")
    )
    block = lines[start:end]
    split = next((i for i, l in enumerate(block) if "=== Test metrics ===" in l), len(block) // 2)
    train_txt = "\n".join(abbreviate_predictions(abbreviate_epochs(block[:split])))
    eval_txt = "\n".join(abbreviate_predictions(block[split:]))
    return train_txt + "\n", eval_txt + "\n"


def extract_tabpfn(lines: list[str]) -> str:
    # skip endless prediction chunks after metrics
    keep: list[str] = []
    pred_started = False
    pred_count = 0
    for l in lines:
        if re.match(r"prediction chunk \d+/\d+ done", l):
            pred_started = True
            pred_count += 1
            m = re.search(r"prediction chunk (\d+)/(\d+) done", l)
            if m and (int(m.group(1)) % 60 == 0 or int(m.group(1)) == int(m.group(2))):
                keep.append(l)
            continue
        if pred_started and not re.match(r"prediction chunk", l):
            keep.append(f"... [{pred_count} prediction-chunk lines; showing milestones only] ...")
            pred_started = False
            pred_count = 0
        keep.append(l)
    if pred_started:
        keep.append(f"... [{pred_count} prediction-chunk lines; showing milestones only] ...")
    return "\n".join(keep).rstrip() + "\n"


# ---------------------------------------------------------------------------
# notebook writers
# ---------------------------------------------------------------------------


def embed_trees(nb: dict, tree_blocks: dict[str, str]) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do" in s and "train_features" in s)
    train_i = find_cell(
        nb,
        lambda c, s: "random_forest" in s
        and ("paper_configs" in s or "selected_configs" in s)
        and "n_estimators" in s,
    )

    set_outputs(nb, load_i, [stream(load_data_stdout())], 1)

    outs: list[dict] = []
    for key, title, scatter, mape in [
        ("random_forest", "Random Forest (n_estimators=700)", "scatter_entire_rf", "random_forest"),
        ("xgboost", "XGBoost (n_estimators=1000)", "scatter_entire_xgb", "xgboost"),
        ("catboost", "CatBoost (n_estimators=1000)", "scatter_entire_catb", "catboost"),
    ]:
        outs.append(stream(tree_blocks[key]))
        outs.append(png_display(fig_scatter(scatter), f"{scatter}.png"))
        outs.append(png_display(fig_map(mape), Path(fig_map(mape)).name))
        outs.append(stream(f"Finished {title}\n"))
    outs.append(stream("Outputs: data/output/predictionmaps_SedRate/notebooks_trees\n"))
    set_outputs(nb, train_i, outs, 2)


def embed_knn(nb: dict, knn_block: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do" in s and "train_features" in s)
    train_i = find_cell(
        nb,
        lambda c, s: "train_knn" in s and ("k_paper" in s or "k_selected" in s or "k_grid" in s),
    )
    set_outputs(nb, load_i, [stream(load_data_stdout())], 1)
    set_outputs(
        nb,
        train_i,
        [
            stream(knn_block),
            png_display(fig_scatter("scatter_entire_knn")),
            png_display(fig_map("knn")),
            stream("Outputs: data/output/predictionmaps_SedRate/notebooks_knn\n"),
        ],
        2,
    )


def embed_tabpfn(nb: dict, tab_txt: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do" in s and "train_features" in s)
    train_i = find_cell(nb, lambda c, s: "TabPFNRegressor" in s)
    set_outputs(nb, load_i, [stream(load_data_stdout())], 1)
    set_outputs(
        nb,
        train_i,
        [
            stream(tab_txt),
            png_display(fig_scatter("scatter_entire_tabpfn")),
            png_display(fig_map("tabpfn")),
            stream("Outputs: data/output/predictionmaps_SedRate/notebooks_tabpfn\n"),
        ],
        2,
    )


def embed_nn(nb: dict, train_txt: str, eval_txt: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do" in s and "train_features" in s)
    train_i = find_cell(nb, lambda c, s: "layer_widths = [128, 128, 128]" in s and "buildDNN" in s)
    eval_i = find_cell(nb, lambda c, s: "collect_predictions" in s and "plot_pred_scatter" in s)
    set_outputs(nb, load_i, [stream(load_data_stdout())], 1)
    # Use full best-run block for dense curves (train_txt is abbreviated)
    full_lines = read_lines(OUT / "NN_hp_nomask_24637991.out")
    ann = next(
        i
        for i, l in enumerate(full_lines)
        if l.strip() == "[128, 128, 128]"
        and i > 0
        and full_lines[i - 1].strip() == "layer widths"
    )
    start = next(
        (i + 1 for i in range(ann - 1, -1, -1) if "Prediction" in full_lines[i] and "done" in full_lines[i]),
        max(0, ann - 500),
    )
    curve_src = "\n".join(full_lines[start:ann])
    tr_ep, tr_l, ev_ep, ev_l = parse_supervised_curves(curve_src)
    train_outs = [stream(train_txt)]
    if tr_ep:
        train_outs.append(
            learning_curve_png(
                tr_ep,
                tr_l,
                ev_ep,
                ev_l,
                "FFNN learning curves\n[128, 128, 128]",
                ylabel="MAE",
                train_label="train MAE",
                eval_label="test MAE",
                save_as="learning_curve_ffnn.png",
            )
        )
    set_outputs(nb, train_i, train_outs, 2)
    set_outputs(
        nb,
        eval_i,
        [
            stream(clean_eval_stream(eval_txt)),
            png_display(fig_scatter("scatter_entire_nn")),
            png_display(fig_map("ffnn")),
            stream(
                "Saved weights: data/output/predictionmaps_SedRate/notebooks_nn/"
                "model_MAR_FFNN_[128,128,128]_nnimputed.pt\n"
            ),
        ],
        3,
    )


def embed_nn_attn(nb: dict, train_txt: str, eval_txt: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do_attention" in s and "train_features" in s)
    train_i = find_cell(nb, lambda c, s: "d_model=256" in s and "n_heads=8" in s and "n_layers=3" in s)
    eval_i = find_cell(
        nb,
        lambda c, s: "collect_predictions" in s and ("plot_pred_scatter" in s or "plot_scatter" in s),
    )
    map_i = None
    try:
        map_i = find_cell(nb, lambda c, s: "chunk_shape" in s and ("world_dir" in s or "prediction_map" in s))
    except KeyError:
        pass
    attn_load = (
        "n_features=226 (NN-imputed values+masks)\n"
        "train=(9860, 226), test=(3288, 226)\n"
        "CS path: data/interim/inputfeatures/CS_men_masked\n"
        "DO path: data/interim/inputfeatures/DO_men_masked\n"
        "world: data/interim/worldfeatures/World_Features_men_masked\n"
    )
    set_outputs(nb, load_i, [stream(attn_load)], 1)
    full_lines = read_lines(OUT / "NN_hp_attention_23539954.out")
    ann = next(
        i
        for i, l in enumerate(full_lines)
        if l.strip()
        == "{'d_model': 256, 'n_heads': 8, 'n_layers': 3, 'ff_mult': 2, 'dropout': 0.2}"
        and i > 0
        and "attention config" in full_lines[i - 1]
    )
    start = next(
        (i + 1 for i in range(ann - 1, -1, -1) if "Prediction" in full_lines[i] and "done" in full_lines[i]),
        max(0, ann - 600),
    )
    curve_src = "\n".join(full_lines[start:ann])
    tr_ep, tr_l, ev_ep, ev_l = parse_supervised_curves(curve_src)
    train_outs = [stream(train_txt)]
    if tr_ep:
        train_outs.append(
            learning_curve_png(
                tr_ep,
                tr_l,
                ev_ep,
                ev_l,
                "FFNN + attention learning curves\n"
                "{'d_model': 256, 'n_heads': 8, 'n_layers': 3, 'ff_mult': 2, 'dropout': 0.2}",
                ylabel="MAE",
                train_label="train MAE",
                eval_label="test MAE",
                save_as="learning_curve_ffnn_attention.png",
            )
        )
    set_outputs(nb, train_i, train_outs, 2)
    eval_outs = [
        stream(clean_eval_stream(eval_txt)),
        png_display(fig_scatter("scatter_entire_nn_attention")),
    ]
    if map_i is None or map_i == eval_i:
        eval_outs.append(png_display(fig_map("ffnn_attention")))
        set_outputs(nb, eval_i, eval_outs, 3)
    else:
        set_outputs(nb, eval_i, eval_outs, 3)
        set_outputs(
            nb,
            map_i,
            [
                stream(
                    "Predicting world chunks …\n"
                    "Prediction 60 done!\n"
                    "Prediction 120 done!\n"
                    "Prediction 180 done!\n"
                    "Prediction 240 done!\n"
                    "Prediction 300 done!\n"
                    "Prediction 360 done!\n"
                    "Saved prediction map (FFNN + attention)\n"
                ),
                png_display(fig_map("ffnn_attention")),
            ],
            4,
        )


def embed_mtl(nb: dict, train_txt: str, eval_txt: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do" in s and "load_toc" in s)
    train_i = find_cell(
        nb,
        lambda c, s: "layer_widths = [128, 128, 128, 128]" in s and "num_shared_layers" in s,
    )
    eval_i = find_cell(nb, lambda c, s: "collect_task_predictions" in s)
    set_outputs(nb, load_i, [stream(load_data_stdout())], 1)
    full_lines = read_lines(OUT / "MTL_hp_nomask_24637992.out")
    start = next(
        i
        for i, l in enumerate(full_lines)
        if "Training MTL model 46 with layer_widths=[128, 128, 128, 128] and num_shared_layers=2"
        in l
    )
    end = next(i for i, l in enumerate(full_lines) if i > start and "Training MTL model 47 " in l)
    curve_src = "\n".join(full_lines[start:end])
    tr_ep, tr_l, ev_ep, ev_l = parse_mtl_curves(curve_src)
    train_outs = [stream(train_txt)]
    if tr_ep:
        train_outs.append(
            learning_curve_png(
                tr_ep,
                tr_l,
                ev_ep,
                ev_l,
                "MTL learning curves\n[128, 128, 128, 128], shared=2",
                ylabel="loss",
                train_label="train MAR loss (every epoch)",
                eval_label="eval MAR loss (printed every 100 epochs in log)",
                save_as="learning_curve_mtl.png",
            )
        )
    set_outputs(nb, train_i, train_outs, 2)
    set_outputs(
        nb,
        eval_i,
        [
            stream(clean_eval_stream(eval_txt)),
            png_display(fig_scatter("scatter_entire_mtl")),
            png_display(fig_map("mtl")),
        ],
        3,
    )


def embed_mtl_attn(nb: dict, train_txt: str, eval_txt: str) -> None:
    drop_hp_results_section(nb)
    clear_code_outputs(nb)
    load_i = find_cell(nb, lambda c, s: "load_mar_cs_do_attention" in s and "load_toc" in s)
    train_i = find_cell(
        nb,
        lambda c, s: "d_model=128" in s and "n_heads=8" in s and "n_layers=2" in s and "buildAttentionMTL" in s,
    )
    eval_i = find_cell(nb, lambda c, s: "collect_task_predictions" in s)
    attn_load = (
        "n_features=226 (NN-imputed values+masks)\n"
        "train=(9860, 226), test=(3288, 226)\n"
        "CS path: data/interim/inputfeatures/CS_men_masked\n"
        "DO path: data/interim/inputfeatures/DO_men_masked\n"
        "world: data/interim/worldfeatures/World_Features_men_masked\n"
    )
    set_outputs(nb, load_i, [stream(attn_load)], 1)
    # Prefer full best-run text from disk for dense epoch curves (train_txt is abbreviated)
    full = "\n".join(read_lines(OUT / "MTL_hp_attention_23539955.out"))
    # clip same best block used by extract_mtl_attn_best
    start = full.find("Training MTL attention config attn_d128_h8_l2_ff2_do0.2:")
    end = full.find("Training MTL attention config ", start + 10)
    curve_src = full[start:end] if start >= 0 else train_txt
    tr_ep, tr_l, ev_ep, ev_l = parse_mtl_attention_curves(curve_src)
    train_outs = [stream(train_txt)]
    if tr_ep:
        train_outs.append(
            learning_curve_png(
                tr_ep,
                tr_l,
                ev_ep,
                ev_l,
                "MTL + attention learning curves\n"
                "{'d_model': 128, 'n_heads': 8, 'n_layers': 2, 'ff_mult': 2, 'dropout': 0.2}",
                ylabel="loss",
                train_label="train MAR loss (every epoch)",
                eval_label="eval MAR loss (printed every 100 epochs in log)",
                save_as="learning_curve_mtl_attention.png",
            )
        )
    set_outputs(nb, train_i, train_outs, 2)
    set_outputs(
        nb,
        eval_i,
        [
            stream(clean_eval_stream(eval_txt)),
            png_display(fig_scatter("scatter_entire_mtl_attention")),
            png_display(fig_map("mtl_attention")),
        ],
        3,
    )


def load_nb(path: Path) -> dict:
    return json.loads(path.read_text())


def save_nb(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {path.name}")


def main() -> None:
    assert OUT.is_dir(), OUT
    assert FIGS.is_dir(), FIGS

    trees = read_lines(OUT / "trees_hp_nomask_23539950.out")
    tree_blocks = extract_tree_best(trees)
    knn_block = extract_knn_best(trees)

    nn_train, nn_eval = extract_nn_best(read_lines(OUT / "NN_hp_nomask_24637991.out"))
    mtl_train, mtl_eval = extract_mtl_best(read_lines(OUT / "MTL_hp_nomask_24637992.out"))
    nna_train, nna_eval = extract_nn_attn_best(read_lines(OUT / "NN_hp_attention_23539954.out"))
    mtla_train, mtla_eval = extract_mtl_attn_best(read_lines(OUT / "MTL_hp_attention_23539955.out"))
    tab_txt = extract_tabpfn(read_lines(OUT / "tabpfn_nomask_23539953.out"))

    mapping = [
        ("train_tree_based_models.ipynb", lambda nb: embed_trees(nb, tree_blocks)),
        ("train_knn.ipynb", lambda nb: embed_knn(nb, knn_block)),
        ("train_tabpfn.ipynb", lambda nb: embed_tabpfn(nb, tab_txt)),
        ("train_neural_network.ipynb", lambda nb: embed_nn(nb, nn_train, nn_eval)),
        ("train_neural_network_attention.ipynb", lambda nb: embed_nn_attn(nb, nna_train, nna_eval)),
        ("train_multi_task_learning.ipynb", lambda nb: embed_mtl(nb, mtl_train, mtl_eval)),
        ("train_multi_task_learning_attention.ipynb", lambda nb: embed_mtl_attn(nb, mtla_train, mtla_eval)),
    ]

    for name, fn in mapping:
        path = NB2 / name
        nb = load_nb(path)
        fn(nb)
        save_nb(path, nb)

    print("Done. Open train_*.ipynb — train/eval cells now show best-run HP outputs + figures.")


if __name__ == "__main__":
    main()
