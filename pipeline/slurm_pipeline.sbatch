#!/bin/bash
# Full pipeline execution on SLURM (mtl-mar).
#
# Submit:  sbatch pipeline/slurm_pipeline.sbatch
#
#SBATCH --job-name=mtlmar_pipe
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96000
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --partition=base

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

PY="${PYTORCH_PYTHON:-${HOME}/.conda/envs/pytorch/bin/python}"

echo "=== Pipeline started: $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "ROOT:   $ROOT"
echo "Python: $PY"

# Fast path: existing interim features, paper tree defaults, no maps
"$PY" pipeline/run_pipeline.py \
    --use-existing-features \
    --skip-hp-sweep \
    --models rf xgb cb knn \
    2>&1

# From scratch (rebuild site features from raw rasters):
# "$PY" pipeline/run_pipeline.py --models rf xgb cb knn 2>&1

# Include DNN (GPU partition recommended):
# "$PY" pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch mlp 2>&1
# "$PY" pipeline/run_pipeline.py --use-existing-features --dnn --dnn-arch attention 2>&1

echo "=== Pipeline finished: $(date) ==="
