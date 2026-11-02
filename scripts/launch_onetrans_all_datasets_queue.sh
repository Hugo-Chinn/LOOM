#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_all_datasets_multiseed"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" scripts/build_onetrans_all_datasets_multiseed.py
"$PYTHON_BIN" scripts/run_onetrans_all_datasets_queue.py --init

for gpu in 2 3 4 5 6 7; do
  pid_file="$RUN_DIR/worker_gpu${gpu}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "worker gpu=${gpu} already running pid=$(cat "$pid_file")"
    continue
  fi
  nohup "$PYTHON_BIN" scripts/run_onetrans_all_datasets_queue.py \
    --gpu "$gpu" --free_threshold_mib 1024 --poll_seconds 30 \
    > "$RUN_DIR/worker_gpu${gpu}.launcher.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "launched worker gpu=${gpu} pid=$!"
done
