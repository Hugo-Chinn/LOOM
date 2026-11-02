#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs_local/onetrans_ns_s_state_multiseed
.venv-unirank/bin/python scripts/run_onetrans_ns_s_state_queue.py --init

pids=()
for gpu in 2 3 4 5 6 7; do
  .venv-unirank/bin/python scripts/run_onetrans_ns_s_state_queue.py \
    --gpu "$gpu" --free_threshold_mib 1024 --poll_seconds 30 \
    >> "runs_local/onetrans_ns_s_state_multiseed/worker_gpu${gpu}.launcher.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done
