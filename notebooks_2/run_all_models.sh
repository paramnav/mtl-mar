#!/usr/bin/env bash
# Submit or run every paper train notebook (trees, kNN, TabPFN, FFNN, MTL, attention).
#
# Usage (from anywhere):
#   bash notebooks_2/run_all_models.sh              # sbatch all jobs
#   bash notebooks_2/run_all_models.sh --local       # run sequentially on this node
#   bash notebooks_2/run_all_models.sh --list        # print jobs only
#   bash notebooks_2/run_all_models.sh --trees-only  # CPU models only
#   bash notebooks_2/run_all_models.sh --gpu-only    # GPU models only
#
# Optional env:
#   PYTORCH_PYTHON=/path/to/python
#   SBATCH_PARTITION_CPU=base
#   SBATCH_PARTITION_GPU=gpu
#
# For new labels / feature grids / NN-imputation first, see:
#   docs/RETRAINING_WITH_NEW_DATA.md
#   notebooks_2/run_models_on_new_data.ipynb
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NB2="$ROOT/notebooks_2"
OUT="$NB2/outfiles"
ERR="$NB2/errfiles"
mkdir -p "$OUT" "$ERR"

PY="${PYTORCH_PYTHON:-${HOME}/.conda/envs/pytorch/bin/python}"
export PATH="$(dirname "$PY"):${PATH:-}"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg

MODE=sbatch   # sbatch | local | list
FILTER=all    # all | trees | gpu

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local) MODE=local ;;
    --list) MODE=list ;;
    --trees-only) FILTER=trees ;;
    --gpu-only) FILTER=gpu ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
done

# name|notebook|variant|cpus|gpus|mem_mb|time|partition_default
JOBS=(
  "trees|train_tree_based_models.ipynb|values_only|8|0|64000|6:00:00|${SBATCH_PARTITION_CPU:-base}"
  "knn|train_knn.ipynb|values_only|4|0|32000|4:00:00|${SBATCH_PARTITION_CPU:-base}"
  "tabpfn|train_tabpfn.ipynb|values_only|4|0|64000|12:00:00|${SBATCH_PARTITION_CPU:-base}"
  "nn|train_neural_network.ipynb|values_only|4|1|64000|24:00:00|${SBATCH_PARTITION_GPU:-gpu}"
  "mtl|train_multi_task_learning.ipynb|values_only|4|1|64000|36:00:00|${SBATCH_PARTITION_GPU:-gpu}"
  "nn_attn|train_neural_network_attention.ipynb|attention|4|1|64000|24:00:00|${SBATCH_PARTITION_GPU:-gpu}"
  "mtl_attn|train_multi_task_learning_attention.ipynb|attention|4|1|64000|36:00:00|${SBATCH_PARTITION_GPU:-gpu}"
)

run_one_local() {
  local name="$1" nb="$2" variant="$3"
  export MASKED_FEATURE_VARIANT="$variant"
  echo "=== $nb ($variant) started: $(date) ==="
  ( cd "$NB2" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.kernel_name=python3 \
      "$nb" )
  echo "=== $nb finished: $(date) ==="
}

submit_one() {
  local name="$1" nb="$2" variant="$3" cpus="$4" gpus="$5" mem="$6" wall="$7" part="$8"
  local script
  script="$(mktemp "$OUT/sbatch_${name}_XXXXXX.sh")"
  {
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=mtlmar_${name}"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --tasks-per-node=1"
    echo "#SBATCH --cpus-per-task=${cpus}"
    if [[ "$gpus" != "0" ]]; then
      echo "#SBATCH --gpus-per-node=${gpus}"
    fi
    echo "#SBATCH --mem=${mem}"
    echo "#SBATCH --time=${wall}"
    echo "#SBATCH --output=${OUT}/train_${name}_%j.out"
    echo "#SBATCH --error=${ERR}/train_${name}_%j.err"
    echo "#SBATCH --partition=${part}"
    echo "set -euo pipefail"
    echo "export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg"
    echo "export MASKED_FEATURE_VARIANT=${variant}"
    echo "export PATH=\"$(dirname "$PY"):\$PATH\""
    echo "cd \"$NB2\""
    echo "echo \"=== $nb started: \$(date) ===\""
    if [[ "$gpus" != "0" ]]; then
      echo "\"$PY\" -c \"import torch; print('cuda', torch.cuda.is_available())\""
    fi
    echo "\"$PY\" -m jupyter nbconvert --to notebook --execute --inplace \\"
    echo "  --ExecutePreprocessor.timeout=86400 --ExecutePreprocessor.kernel_name=python3 \\"
    echo "  \"$nb\""
    echo "echo \"=== finished: \$(date) ===\""
  } > "$script"
  chmod +x "$script"
  sbatch "$script"
}

echo "mtl-mar root: $ROOT"
echo "python:       $PY"
echo "mode:         $MODE  filter=$FILTER"
echo

for spec in "${JOBS[@]}"; do
  IFS='|' read -r name nb variant cpus gpus mem wall part <<< "$spec"
  if [[ "$FILTER" == "trees" && "$gpus" != "0" ]]; then
    continue
  fi
  if [[ "$FILTER" == "gpu" && "$gpus" == "0" ]]; then
    continue
  fi
  if [[ ! -f "$NB2/$nb" ]]; then
    echo "MISSING $nb — skip" >&2
    continue
  fi
  case "$MODE" in
    list)
      echo "$name  $nb  variant=$variant  gpus=$gpus  partition=$part"
      ;;
    local)
      run_one_local "$name" "$nb" "$variant"
      ;;
    sbatch)
      submit_one "$name" "$nb" "$variant" "$cpus" "$gpus" "$mem" "$wall" "$part"
      ;;
  esac
done

if [[ "$MODE" == "sbatch" ]]; then
  echo
  echo "Submitted. Monitor with:  tail -f $OUT/train_*_*.out"
fi
echo
echo "Note: these jobs use the selected paper hyperparameters (Table 2)."
echo "Tree/KNN hyperparameter sweeps: python pipeline/run_pipeline.py"
echo "Full new-data workflow: docs/RETRAINING_WITH_NEW_DATA.md"
