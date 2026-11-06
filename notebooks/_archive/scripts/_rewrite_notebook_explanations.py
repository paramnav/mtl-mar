#!/usr/bin/env python3
"""Rewrite train_*.ipynb markdown + comments in the style of MTL+attention.

Preserves existing cell outputs. Run once, then optionally re-run
embed_hp_results_into_notebooks.py to refresh figures/curves.
"""
from __future__ import annotations

import json
from pathlib import Path

NB2 = Path(__file__).resolve().parent

METRICS_HELPER = '''def metrics(y_true, y_pred):
    """Common regression skill scores for continuous targets (usually MAR).

    mae  — mean absolute error (same units as the target; lower is better)
    corr — Pearson correlation between observations and predictions
    r2   — coefficient of determination (1 = perfect; can be negative)
    """
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "corr": float(np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]),
        "r2": float(r2_score(y_true, y_pred)),
    }


def plot_pred_scatter(y_train, train_pred, y_test, test_pred, title):
    """Observed vs predicted scatter for train and test sites.

    The dashed line is the 1:1 relation: predictions near it are accurate.
    """
    tr = metrics(y_train, train_pred)
    te = metrics(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_train, train_pred, s=8, alpha=0.35, label=f"train MAE={tr['mae']:.2f}, r={tr['corr']:.3f}")
    ax.scatter(y_test, test_pred, s=8, alpha=0.45, label=f"test MAE={te['mae']:.2f}, r={te['corr']:.3f}")
    lims = [
        min(y_train.min(), y_test.min(), train_pred.min(), test_pred.min()),
        max(y_train.max(), y_test.max(), train_pred.max(), test_pred.max()),
    ]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()
    return tr, te
'''

LOAD_VALUES_MD = '''## Load MAR observations and environmental predictors

We assemble paired **features** (environmental predictors) and **labels**
(observed mass accumulation rate, MAR) from Coastal Sediment (CS) and Deep Ocean
(DO) core compilations.

Features are **neural-network imputed values only** (113 dimensions): originally
missing predictors were filled by a separate imputation model, then the continuous
values are used here. Sites are split **75% train / 25% test**, stratified by MAR
deciles so high- and low-MAR regimes appear in both sets.
'''

LOAD_VALUES_CODE = '''# Load CS + DO MAR sites with a stratified train/test split.
# Returns feature matrices, labels, and the mean/std used later to standardize
# global ocean grids the same way as the training data.
mar = load_mar_cs_do(preprocess_features_labels)

train_features = mar["train_features"]  # predictors at training sites
train_labels = mar["train_labels"]      # observed MAR at training sites
test_features = mar["test_features"]    # held-out predictors
test_labels = mar["test_labels"]        # held-out MAR (for skill scores)
X_mean, X_std = mar["x_mean"], mar["x_std"]  # standardization from the train set
world_dir = mar["world_dir"]            # folder of global ocean feature chunks

print(f"n_features={mar['n_features']} (NN-imputed values only)")
print(f"train={train_features.shape}, test={test_features.shape}")
print(f"CS path: {mar['paths']['CS']}")
print(f"DO path: {mar['paths']['DO']}")
print(f"world: {world_dir}")
'''


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return text.splitlines(keepends=True)


def set_src(nb: dict, idx: int, text: str) -> None:
    nb["cells"][idx]["source"] = lines(text)


def load_nb(name: str) -> dict:
    return json.loads((NB2 / name).read_text())


def save_nb(name: str, nb: dict) -> None:
    (NB2 / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print("updated", name)


def rewrite_mtl() -> None:
    name = "train_multi_task_learning.ipynb"
    nb = load_nb(name)
    set_src(nb, 0, METRICS_HELPER)
    set_src(
        nb,
        1,
        '''## Load paired MAR and TOC observations

We train one network on two related sediment properties at the same locations:

- **MAR** — mass accumulation rate (primary target)
- **TOC** — total organic carbon (auxiliary task that shares representation with MAR)

Features are the 113-d neural-network–imputed environmental predictors (values only).
The train/test split is stratified by MAR deciles.
''',
    )
    set_src(
        nb,
        2,
        '''# Load CS + DO cores for both tasks with a shared geographic train/test split.
mar = load_mar_cs_do(preprocess_features_labels)
toc = load_toc_cs_do(preprocess_features_labels)

# MAR (sedimentation / mass-accumulation) matrices
sedrate_train_features = mar["train_features"]
sedrate_train_labels = mar["train_labels"]
sedrate_test_features = mar["test_features"]
sedrate_test_labels = mar["test_labels"]
sedrate_features = mar["features"]  # all sites — used to size the network

# TOC matrices — same sites, different response variable
toc_train_features = toc["train_features"]
toc_train_labels = toc["train_labels"]
toc_test_features = toc["test_features"]
toc_test_labels = toc["test_labels"]

# Standardization stats for later global-grid prediction
X_mean, X_std = mar["x_mean"], mar["x_std"]
world_dir = mar["world_dir"]

print("MAR train/test:", sedrate_train_features.shape, sedrate_test_features.shape)
print("TOC train/test:", toc_train_features.shape, toc_test_features.shape)
print("n_features:", mar["n_features"])
''',
    )
    set_src(
        nb,
        3,
        '''## Mini-batch iterators and loss function

Training proceeds in **mini-batches**: small random subsets of sites are shown
to the model in each step. Separate loaders for MAR and TOC let the multi-task
trainer alternate updates on both heads.

The objective is **Huber loss**, which behaves like squared error for small
residuals and like absolute error for large residuals — useful for heavy-tailed MAR.
''',
    )
    set_src(
        nb,
        4,
        '''# Number of sites shown to the network in one optimization step
batch_size = 32

# Training loaders: shuffle=True randomizes site order each epoch
train_dataloaders = {
    "sedimentation_rates": DataLoader(
        TensorDataset(
            torch.tensor(sedrate_train_features, dtype=torch.float32),
            torch.tensor(sedrate_train_labels, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    ),
    "toc": DataLoader(
        TensorDataset(
            torch.tensor(toc_train_features, dtype=torch.float32),
            torch.tensor(toc_train_labels, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    ),
}

# Test loaders: shuffle=False keeps evaluation deterministic
test_dataloaders = {
    "sedimentation_rates": DataLoader(
        TensorDataset(
            torch.tensor(sedrate_test_features, dtype=torch.float32),
            torch.tensor(sedrate_test_labels, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    ),
    "toc": DataLoader(
        TensorDataset(
            torch.tensor(toc_test_features, dtype=torch.float32),
            torch.tensor(toc_test_labels, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    ),
}

# Robust regression loss (less sensitive to extreme MAR outliers than MSE alone)
loss_fn = torch.nn.HuberLoss()
''',
    )
    set_src(
        nb,
        5,
        '''## Fit the multi-task feed-forward network

This cell builds and trains a **multi-task learning (MTL)** model:

1. Shared hidden layers learn a joint representation of the environmental predictors.
2. Task-specific heads then predict **MAR** and **TOC** from that representation.

Architecture below was selected by a held-out hyperparameter search: four hidden
layers of width 128 with **two shared layers** (map configuration 47 in the search
log). Training uses early stopping on validation MAR loss; the learning curves
show whether the fit is stable or overfits.
''',
    )
    set_src(
        nb,
        6,
        '''# Architecture from hyperparameter search (held-out MAR skill):
#   layer_widths      — hidden-layer sizes of the shared trunk + heads
#   num_shared_layers — how many leading layers are shared by MAR and TOC
layer_widths = [128, 128, 128, 128]
num_shared_layers = 2
num_epochs = 2000  # upper bound; early stopping usually ends sooner

train_step = 2     # alternate MAR / TOC updates inside the multi-task trainer
write_maps = True  # if True, later cell writes global ocean prediction grids
count_mtl = 47     # search-log id for this architecture (used in output filenames)

# Fix random seeds so weight initialization is reproducible
set_seed(42)
model = buildDNN(layer_widths, sedrate_features, num_shared_layers)
model.double()  # float64 — matches the numerical precision used in the search

# Optimize network weights. Returns per-epoch multi-task losses and the LR schedule.
training_losses, learning_rate, evaluation_losses = train_model(
    model,
    loss_fn,
    train_dataloaders,
    test_dataloaders,
    train_step=train_step,
    num_epochs=num_epochs,
)

out_dir = PREDICTION_MAPS_DIR / "notebooks_mtl"
out_dir.mkdir(parents=True, exist_ok=True)

# Helper: pull the MAR entry out of a multi-task loss dict for plotting
def _sed_loss(x):
    return x["sedimentation_rates"] if isinstance(x, dict) else float(x)

# Learning curves: falling train+eval → improving fit; rising eval alone → overfitting
train_curve = [_sed_loss(v) for v in training_losses]
eval_curve = [_sed_loss(v) for v in evaluation_losses]
plt.figure(figsize=(7, 3))
plt.plot(train_curve, label="train MAR loss")
plt.plot(eval_curve, label="eval MAR loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.title(f"MTL learning curves\\n{layer_widths}, shared={num_shared_layers}")
plt.tight_layout()
plt.show()  # figure is shown below this cell
''',
    )
    set_src(
        nb,
        7,
        '''## Evaluate predictive skill and map global fields

With the fitted network we:

1. Score **train** and **test** sites for both MAR and TOC (MAE, correlation, $R^2$).
2. Draw observed-vs-predicted scatter plots.
3. Save network weights for reuse.
4. Optionally apply the model to the global ocean feature grid to produce continuous
   MAR and TOC prediction maps (land cells zeroed with an island mask).
''',
    )
    set_src(
        nb,
        8,
        '''# --- helpers -----------------------------------------------------------------

def collect_task_predictions(model, loader, task):
    """Run the network on every batch and collect labels + predictions.

    `task` selects which head to read: "sedimentation_rates" (MAR) or "toc".
    """
    model.eval()  # turn off dropout / training-only behaviour
    ys, ps = [], []
    with torch.no_grad():  # evaluation does not update weights
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float64)
            _decoded, out = model(x, task_id=task)
            ys.append(y.numpy().reshape(-1))
            ps.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)


# --- skill scores + scatter plots for both tasks -----------------------------

rows = []
for task, label in [("sedimentation_rates", "MAR"), ("toc", "TOC")]:
    y_tr, p_tr = collect_task_predictions(model, train_dataloaders[task], task)
    y_te, p_te = collect_task_predictions(model, test_dataloaders[task], task)
    tr, te = plot_pred_scatter(y_tr, p_tr, y_te, p_te, f"MTL — {label}")
    rows.append({"task": label, **{f"test_{k}": v for k, v in te.items()}})

pd.DataFrame(rows).to_csv(out_dir / "mtl_test_summary.csv", index=False)

arch = str(layer_widths).replace(" ", "")
weight_path = out_dir / f"model_MTL_map{count_mtl}_{arch}_shared{num_shared_layers}_nnimputed.pt"
torch.save(model.state_dict(), weight_path)
print("Saved weights:", weight_path)


# --- optional: global ocean prediction maps ----------------------------------

if write_maps:
    chunk_shape = (6, 4320)  # spatial layout of one stored world-feature file
    files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
    map_mar = map_toc = None
    model.eval()
    with torch.no_grad():
        for i, f in enumerate(files):
            feats = normalize_world_chunk(np.load(f), X_mean, X_std)
            t = torch.tensor(feats, dtype=torch.float64, device=device)
            _, p_mar = model(t, task_id="sedimentation_rates")
            _, p_toc = model(t, task_id="toc")
            p_mar = p_mar.detach().cpu().numpy().reshape(chunk_shape)
            p_toc = p_toc.detach().cpu().numpy().reshape(chunk_shape)
            map_mar = p_mar if map_mar is None else np.append(map_mar, p_mar, axis=0)
            map_toc = p_toc if map_toc is None else np.append(map_toc, p_toc, axis=0)
            if (i + 1) % 60 == 0:
                print(f"  predicted {i+1}/{len(files)} chunks")
            del t, feats
            gc.collect()

    land = np.load(DATA_RAW / "island_map.npy")
    land = np.where(np.isnan(land), 1, land).astype(bool)
    for arr, name in [(map_mar, "MAR"), (map_toc, "toc")]:
        arr = np.rot90(np.rot90(np.fliplr(arr)))
        arr[land] = 0
        path = out_dir / f"prediction_map_{name}_entire_MTL_map{count_mtl}_nnimputed.npy"
        np.save(path, arr)
        print("Saved:", path)
''',
    )
    save_nb(name, nb)


def rewrite_nn() -> None:
    name = "train_neural_network.ipynb"
    nb = load_nb(name)
    set_src(nb, 0, METRICS_HELPER)
    set_src(nb, 1, LOAD_VALUES_MD)
    set_src(nb, 2, LOAD_VALUES_CODE)
    set_src(
        nb,
        3,
        '''## Mini-batch iterators and loss function

The feed-forward network is trained with mini-batches of sites. **Huber loss**
is used as the objective so large MAR outliers do not dominate the fit.
''',
    )
    set_src(
        nb,
        4,
        '''batch_size = 64  # sites per optimization step

# Pair each feature vector with its MAR label; shuffle training order each epoch
train_ds = TensorDataset(
    torch.tensor(train_features, dtype=torch.float64),
    torch.tensor(train_labels, dtype=torch.float64),
)
test_ds = TensorDataset(
    torch.tensor(test_features, dtype=torch.float64),
    torch.tensor(test_labels, dtype=torch.float64),
)
trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Robust regression loss (less sensitive to extreme MAR outliers than MSE alone)
loss_fn = torch.nn.HuberLoss()
''',
    )
    set_src(
        nb,
        5,
        '''## Fit the supervised feed-forward network

A fully connected network maps the 113 environmental predictors to MAR.
Hidden-layer widths `[128, 128, 128]` were selected by a held-out hyperparameter
search. Training uses Adam with weight decay and early stopping; the learning
curve (train vs test MAE) below diagnoses underfitting / overfitting.
''',
    )
    set_src(
        nb,
        6,
        '''# Architecture from hyperparameter search (held-out MAR skill)
layer_widths = [128, 128, 128]
num_epochs = 1800  # upper bound; early stopping usually ends sooner

train_step = 2
write_maps = True  # if True, later cell writes a global MAR prediction map

features_all = mar["features"]
labels_all = mar["labels"]

set_seed(42)  # reproducible weight initialization
model = buildDNN(layer_widths, features_all)
model.apply(initialize_weights)
model.double()
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

# Fit weights; returns per-epoch MAE curves used for the plot below
training_accuracy, training_losses, training_mae, learning_rate, evaluation_losses, evaluation_mae, true_labels, pred_labels = train_model(
    model, loss_fn, trainloader, testloader, num_epochs=num_epochs,
    lr=1e-3, weight_decay=1e-4,  # matches HP-search run for [128,128,128]
)

out_dir = PREDICTION_MAPS_DIR / "notebooks_nn"
out_dir.mkdir(parents=True, exist_ok=True)

# Learning curves: falling train+test MAE → improving fit
plt.figure(figsize=(7, 3))
plt.plot(training_mae, label="train MAE")
plt.plot(evaluation_mae, label="test MAE")
plt.xlabel("epoch")
plt.ylabel("MAE")
plt.legend()
plt.title(f"FFNN learning curves\\n{layer_widths}")
plt.tight_layout()
plt.show()  # figure is shown below this cell
''',
    )
    set_src(
        nb,
        7,
        '''## Evaluate predictive skill and map the global ocean

Score train/test sites, draw an observed-vs-predicted scatter, save weights, and
optionally predict MAR on the global feature grid (chunk by chunk) with land masked.
''',
    )
    set_src(
        nb,
        8,
        '''def collect_predictions(model, loader):
    """Run the network on every batch and collect labels + predictions."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float64)
            out = model(x)
            ys.append(y.numpy().reshape(-1))
            ps.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)

y_tr, p_tr = collect_predictions(model, trainloader)
y_te, p_te = collect_predictions(model, testloader)
tr, te = plot_pred_scatter(y_tr, p_tr, y_te, p_te, f"FFNN {layer_widths}")
pd.DataFrame([{"model": "ffnn", "layer_widths": str(layer_widths), **{f"test_{k}": v for k, v in te.items()}}]).to_csv(
    out_dir / "nn_test_summary.csv", index=False
)

# Save weights for reuse / map reproduction
arch = str(layer_widths).replace(" ", "")
weight_path = out_dir / f"model_MAR_FFNN_{arch}_nnimputed.pt"
torch.save(model.state_dict(), weight_path)
print("Saved weights:", weight_path)

if write_maps:
    chunk_shape = (6, 4320)  # spatial layout of one stored world-feature file
    files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
    prediction_map = None
    model.eval()
    with torch.no_grad():
        for i, f in enumerate(files):
            feats = normalize_world_chunk(np.load(f), X_mean, X_std)
            t = torch.tensor(feats, dtype=torch.float64, device=device)
            preds = model(t).detach().cpu().numpy().reshape(chunk_shape)
            prediction_map = preds if prediction_map is None else np.append(prediction_map, preds, axis=0)
            if (i + 1) % 60 == 0:
                print(f"  predicted {i+1}/{len(files)} chunks")
            del t, feats
            gc.collect()
    land = np.load(DATA_RAW / "island_map.npy")
    land = np.where(np.isnan(land), 1, land).astype(bool)
    prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))
    prediction_map[land] = 0
    map_path = out_dir / f"prediction_map_MAR_entire_FFNN_{arch}_nnimputed.npy"
    np.save(map_path, prediction_map)
    print("Saved map:", map_path)
''',
    )
    save_nb(name, nb)


def rewrite_nn_attn() -> None:
    name = "train_neural_network_attention.ipynb"
    nb = load_nb(name)
    set_src(
        nb,
        0,
        '''## Load MAR observations (attention features)

Each sample is a **226-dimensional** vector: 113 environmental predictor **values**
plus 113 binary **presence/absence masks** that mark which predictors were originally
missing before imputation. The attention encoder uses both channels so the model can
learn which inputs are observed versus imputed.

The train/test split is stratified by MAR deciles (75%/25%).
''',
    )
    set_src(
        nb,
        1,
        '''# Attention loaders return value+mask features (226 columns)
mar = load_mar_cs_do_attention()
train_features = mar["train_features"]
train_labels = mar["train_labels"]
test_features = mar["test_features"]
test_labels = mar["test_labels"]
X_mean, X_std = mar["x_mean"], mar["x_std"]
world_dir = mar["world_dir"]
n_value_features = mar["n_value_features"]  # 113 value channels; masks follow
features_all = mar["features"]

print(f"n_features={mar['n_features']} (113 values + 113 masks)")
print(f"train={train_features.shape}, test={test_features.shape}")
print(f"CS={mar['paths']['CS']}")
print(f"DO={mar['paths']['DO']}")
print(f"world={world_dir}")
assert mar["n_features"] == 226
''',
    )
    set_src(
        nb,
        2,
        '''## Mini-batch iterators and loss function

Sites are fed to the network in mini-batches. **Huber loss** limits the influence
of extreme MAR outliers during training.
''',
    )
    set_src(
        nb,
        3,
        '''batch_size = 64  # sites per optimization step

trainloader = DataLoader(
    TensorDataset(
        torch.tensor(train_features, dtype=torch.float64),
        torch.tensor(train_labels, dtype=torch.float64),
    ),
    batch_size=batch_size,
    shuffle=True,   # randomize training order each epoch
)
testloader = DataLoader(
    TensorDataset(
        torch.tensor(test_features, dtype=torch.float64),
        torch.tensor(test_labels, dtype=torch.float64),
    ),
    batch_size=batch_size,
    shuffle=False,  # deterministic evaluation
)
loss_fn = torch.nn.HuberLoss()
print("loss: HuberLoss; batch_size:", batch_size)
''',
    )
    set_src(
        nb,
        4,
        '''## Fit the attention-based feed-forward network

A **feature-token attention** encoder treats each of the 113 predictors as a token
(value + mask), mixes information across predictors with self-attention, and then
regresses MAR. Hyperparameters below were selected by a held-out search
(`d_model=256`, 8 heads, 3 layers, feed-forward multiplier 2, dropout 0.2).
''',
    )
    set_src(
        nb,
        5,
        '''# Architecture from hyperparameter search (held-out MAR skill):
#   d_model  — width of each feature token after embedding
#   n_heads  — parallel attention heads (must divide d_model)
#   n_layers — stacked self-attention + feed-forward blocks
#   ff_mult  — feed-forward hidden width = ff_mult * d_model
#   dropout  — regularization (fraction of units dropped in training)
attn_cfg = dict(d_model=256, n_heads=8, n_layers=3, ff_mult=2, dropout=0.2)
num_epochs = 1800  # upper bound; early stopping usually ends sooner
out_dir = PREDICTION_MAPS_DIR / "notebooks_nn_attention"
out_dir.mkdir(parents=True, exist_ok=True)

set_seed(42)
model = buildAttentionDNN(
    features_all,
    n_value_features=n_value_features,
    **attn_cfg,
)
model.double()
if torch.cuda.is_available():
    model = model.cuda()
    loss_fn = loss_fn.cuda()

print("Starting training ...", attn_cfg)
(
    training_accuracy,
    training_losses,
    training_mae,
    learning_rate,
    evaluation_losses,
    evaluation_mae,
    true_labels,
    pred_labels,
) = train_model(
    model,
    loss_fn,
    trainloader,
    testloader,
    num_epochs=num_epochs,
    lr=1e-4,
    weight_decay=1e-4,
)
print(f"Finished training after {len(training_mae)} epochs (early-stop possible)")

# Learning curves diagnose underfitting / overfitting
plot_learning_curves(
    training_mae,
    evaluation_mae,
    f"FFNN + attention learning curves\\n{attn_cfg}",
    save_path=out_dir / "learning_curve_ffnn_attention.png",
)
''',
    )
    set_src(
        nb,
        6,
        '''## Evaluate predictive skill

Score train and test sites (MAE, correlation, $R^2$), draw an observed-vs-predicted
scatter, and save the fitted weights.
''',
    )
    set_src(
        nb,
        7,
        '''def collect_predictions(model, loader):
    """Run the network on every batch and collect labels + predictions."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float64)
            out = model(x)
            ys.append(y.numpy().reshape(-1))
            ps.append(out.detach().cpu().numpy().reshape(-1))
    return np.concatenate(ys), np.concatenate(ps)

y_tr, p_tr = collect_predictions(model, trainloader)
y_te, p_te = collect_predictions(model, testloader)
tr, te = plot_scatter(
    y_tr, p_tr, y_te, p_te,
    f"FFNN + attention — {attn_cfg}",
    save_path=out_dir / "scatter_ffnn_attention.png",
)
compare_table2("ffnn_attention", te["mae"], te["corr"])
pd.DataFrame([{
    "model": "ffnn_attention",
    **{f"cfg_{k}": v for k, v in attn_cfg.items()},
    **{f"test_{k}": v for k, v in te.items()},
    **{f"train_{k}": v for k, v in tr.items()},
}]).to_csv(out_dir / "nn_attention_test_summary.csv", index=False)

# Encode architecture in the filename so different search configs do not overwrite
tag = (
    f"attn_d{attn_cfg['d_model']}_h{attn_cfg['n_heads']}"
    f"_l{attn_cfg['n_layers']}_ff{attn_cfg['ff_mult']}_do{attn_cfg['dropout']}"
)
weight_path = out_dir / f"model_MAR_FFNN_attention_{tag}_nnimputed.pt"
torch.save(model.state_dict(), weight_path)
print("Saved weights:", weight_path)
''',
    )
    set_src(
        nb,
        8,
        '''## Map MAR across the global ocean

World features are stored as many small `.npy` chunks. We predict each chunk,
stitch them into a full map, orient it north-up, and save both array and figure.
''',
    )
    # keep existing map cell logic but add comments — read current and wrap if needed
    map_src = "".join(nb["cells"][9].get("source", []))
    if "spatial layout" not in map_src:
        set_src(
            nb,
            9,
            '''chunk_shape = (6, 4320)  # spatial layout of one stored world-feature file
files = sorted(p for p in world_dir.iterdir() if p.suffix == ".npy")
print(f"Predicting {len(files)} world chunks from {world_dir}")

prediction_map = None
model.eval()
with torch.no_grad():
    for i, f in enumerate(files):
        # Standardize with the same mean/std used at training time
        feats = normalize_world_chunk_attention(np.load(f), X_mean, X_std)
        t = torch.tensor(feats, dtype=torch.float64, device=device)
        preds = model(t).detach().cpu().numpy().reshape(chunk_shape)
        prediction_map = preds if prediction_map is None else np.append(prediction_map, preds, axis=0)
        if (i + 1) % 20 == 0 or (i + 1) == len(files):
            print(f"  chunk {i+1}/{len(files)}  running shape={prediction_map.shape}")
        del t, feats
        gc.collect()

# Orient to north-up / island_map (same rotation as other NN pipelines)
prediction_map = np.rot90(np.rot90(np.fliplr(prediction_map)))
map_npy = out_dir / f"prediction_map_MAR_entire_FFNN_attention_{tag}.npy"
map_png = out_dir / f"prediction_map_MAR_entire_FFNN_attention_{tag}.png"
save_and_plot_mar_map(
    prediction_map,
    map_npy,
    map_png,
    title=f"FFNN + attention ({tag})",
    already_north_up=True,
)
print("Done. Outputs in", out_dir)
''',
        )
    save_nb(name, nb)


def rewrite_trees() -> None:
    name = "train_tree_based_models.ipynb"
    nb = load_nb(name)
    set_src(nb, 0, METRICS_HELPER)
    set_src(nb, 1, LOAD_VALUES_MD)
    set_src(nb, 2, LOAD_VALUES_CODE)
    set_src(
        nb,
        3,
        '''## Fit ensemble tree models

We compare three tree-based regressors for MAR:

- **Random Forest** — bagged decision trees
- **XGBoost** — gradient-boosted trees
- **CatBoost** — ordered-boosting trees

The number of trees (`n_estimators`) for each model was selected by a held-out
hyperparameter search (700 / 1000 / 1000). Set `run_hyperparameter_search = True`
below to re-sweep those grids instead of using the selected counts.
''',
    )
    set_src(
        nb,
        4,
        '''run_hyperparameter_search = False  # True → sweep n_estimators grids below
write_maps = True                   # write global MAR maps for each model

# Counts selected by held-out search (lowest test MAE in each family)
selected_configs = {
    "random_forest": 700,
    "xgboost": 1000,
    "catboost": 1000,
}
search_grids = {
    "random_forest": [50, 100, 200, 400, 700, 800, 1000],
    "xgboost": [100, 200, 400, 800, 1000],
    "catboost": [100, 200, 400, 800, 1000],
}

out_dir = PREDICTION_MAPS_DIR / "notebooks_trees"
out_dir.mkdir(parents=True, exist_ok=True)
summary_rows = []

for model_type, n_default in selected_configs.items():
    if run_hyperparameter_search:
        # Try several tree counts; keep the one with lowest test MAE
        rows = []
        for n in search_grids[model_type]:
            _, train_mae, test_mae, _, _ = train_tree_based_model(
                train_features, train_labels, test_features, test_labels,
                model_type=model_type, n_estimators=int(n), random_seed=42,
            )
            rows.append({"model": model_type, "n_estimators": int(n), "train_mae": train_mae, "test_mae": test_mae})
            print(f"{model_type} n={n}: train MAE={train_mae:.3f}, test MAE={test_mae:.3f}")
        search_df = pd.DataFrame(rows).sort_values("test_mae")
        search_df.to_csv(out_dir / f"{model_type}_search.csv", index=False)
        n_estimators = int(search_df.iloc[0]["n_estimators"])
        print(f"Best {model_type}: n_estimators={n_estimators}")
    else:
        n_estimators = int(n_default)

    # Fit the selected configuration and score train/test sites
    model, train_mae, test_mae, train_pred, test_pred = train_tree_based_model(
        train_features, train_labels, test_features, test_labels,
        model_type=model_type, n_estimators=n_estimators, random_seed=42,
    )
    tr, te = plot_pred_scatter(
        train_labels, train_pred, test_labels, test_pred,
        f"{model_type} (n={n_estimators})",
    )
    summary_rows.append({
        "model": model_type,
        "n_estimators": n_estimators,
        **{f"test_{k}": v for k, v in te.items()},
    })

    if write_maps:
        # Apply the fitted model to the global ocean feature grid
        predict_global_map_sklearn(
            model,
            world_dir,
            X_mean,
            X_std,
            DATA_RAW / "island_map.npy",
            out_dir / f"prediction_map_MAR_entire_{model_type}_n{n_estimators}_nnimputed.npy",
        )

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(out_dir / "tree_models_test_summary.csv", index=False)
print(summary_df)
print("Outputs:", out_dir)
''',
    )
    save_nb(name, nb)


def rewrite_knn() -> None:
    name = "train_knn.ipynb"
    nb = load_nb(name)
    set_src(nb, 0, METRICS_HELPER)
    set_src(nb, 1, LOAD_VALUES_MD)
    set_src(nb, 2, LOAD_VALUES_CODE)
    set_src(
        nb,
        3,
        '''## Fit a k-nearest neighbours regressor

**k-NN** predicts MAR at a site as the average MAR of its *k* most similar training
sites in feature space. The neighbour count `k = 6` was selected by a held-out
search. Set `run_hyperparameter_search = True` to re-sweep `k` instead.
''',
    )
    set_src(
        nb,
        4,
        '''run_hyperparameter_search = False  # True → sweep k instead of using the selected value
write_maps = True
k_selected = 6  # from held-out search (lowest test MAE)
k_grid = [3, 4, 5, 6, 7, 8, 9, 11, 15, 21]

out_dir = PREDICTION_MAPS_DIR / "notebooks_knn"
out_dir.mkdir(parents=True, exist_ok=True)

if run_hyperparameter_search:
    rows = []
    for k in k_grid:
        _, train_mae, test_mae, _, _ = train_knn(
            train_features, train_labels, test_features, test_labels, n_neighbors=k
        )
        rows.append({"k": k, "train_mae": train_mae, "test_mae": test_mae})
        print(f"k={k}: train MAE={train_mae:.3f}, test MAE={test_mae:.3f}")
    search_df = pd.DataFrame(rows).sort_values("test_mae")
    search_df.to_csv(out_dir / "knn_search.csv", index=False)
    k = int(search_df.iloc[0]["k"])
    print("Best k:", k)
else:
    k = k_selected

# Fit k-NN and score train/test sites
model, train_mae, test_mae, train_pred, test_pred = train_knn(
    train_features, train_labels, test_features, test_labels, n_neighbors=k
)
tr, te = plot_pred_scatter(train_labels, train_pred, test_labels, test_pred, f"KNN (k={k})")
pd.DataFrame([{"model": "knn", "k": k, **{f"test_{kk}": vv for kk, vv in te.items()}}]).to_csv(
    out_dir / "knn_test_summary.csv", index=False
)

if write_maps:
    # Apply the fitted model to the global ocean feature grid
    predict_global_map_sklearn(
        model,
        world_dir,
        X_mean,
        X_std,
        DATA_RAW / "island_map.npy",
        out_dir / f"prediction_map_MAR_entire_knn_k{k}_nnimputed.npy",
    )
print("Outputs:", out_dir)
''',
    )
    save_nb(name, nb)


def rewrite_tabpfn() -> None:
    name = "train_tabpfn.ipynb"
    nb = load_nb(name)
    set_src(nb, 0, METRICS_HELPER)
    set_src(nb, 1, LOAD_VALUES_MD)
    set_src(nb, 2, LOAD_VALUES_CODE)
    set_src(
        nb,
        3,
        '''## Fit TabPFN and evaluate

**TabPFN** is a pretrained tabular foundation model used here as a strong baseline
regressor for MAR. It does not require a custom architecture search: we fit it on
the training sites, score the held-out test sites, and optionally map the global ocean.
''',
    )
    set_src(
        nb,
        4,
        '''from tabpfn import TabPFNRegressor

write_maps = True
out_dir = PREDICTION_MAPS_DIR / "notebooks_tabpfn"
out_dir.mkdir(parents=True, exist_ok=True)

# Fit on training sites, then predict train and test MAR
model = TabPFNRegressor()
model.fit(train_features, train_labels)
train_pred = model.predict(train_features)
test_pred = model.predict(test_features)
tr, te = plot_pred_scatter(train_labels, train_pred, test_labels, test_pred, "TabPFN")
pd.DataFrame([{"model": "tabpfn", **{f"test_{k}": v for k, v in te.items()}}]).to_csv(
    out_dir / "tabpfn_test_summary.csv", index=False
)

if write_maps:
    # Chunked prediction (TabPFN can be memory-heavy; process world files one by one)
    predict_global_map_sklearn(
        model,
        world_dir,
        X_mean,
        X_std,
        DATA_RAW / "island_map.npy",
        out_dir / "prediction_map_MAR_entire_tabpfn_nnimputed.npy",
    )
print("Outputs:", out_dir)
''',
    )
    save_nb(name, nb)


def main() -> None:
    rewrite_mtl()
    rewrite_nn()
    rewrite_nn_attn()
    rewrite_trees()
    rewrite_knn()
    rewrite_tabpfn()
    print("Done rewriting explanations.")


if __name__ == "__main__":
    main()
