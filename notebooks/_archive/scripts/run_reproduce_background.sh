#!/usr/bin/env bash
# Launch paper Table-2 reproduction in the background (long CPU job).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT}/notebooks/reproduce_paper_table2.log"
PIDFILE="${ROOT}/notebooks/reproduce_paper_table2.pid"
PY="${PYTORCH_PYTHON:-${HOME}/.conda/envs/pytorch/bin/python}"

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running: PID=$(cat "$PIDFILE")"
  echo "Log: $LOG"
  exit 0
fi

cd "$ROOT"
nohup "$PY" -u notebooks/reproduce_paper_table2.py >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
disown || true
echo "Started PID=$(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Monitor: tail -f $LOG"
echo "When done, results in:"
echo "  data/output/predictionmaps_SedRate/notebooks_paper_table2/"
