#!/usr/bin/env bash
# Queue the four-seed multi-task baseline sweep across the local GPUs.
# Usage: run_mtl_baselines_four_seed.sh [num_parallel_gpus] [gpu_start_index] [expid_filter]
set -uo pipefail
cd "$(dirname "$0")/.."

NPROC="${1:-8}"
GPU_START="${2:-0}"
FILTER="${3:-}"

JOBS="./logs/mtl_baselines_jobs${FILTER:+_${FILTER}}.txt"

python3 - "${NPROC}" "${GPU_START}" "${FILTER}" <<'PY' > "${JOBS}"
import sys
nproc, start, filt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
expids = open("config_mtl_baselines_four_seed/expids.txt").read().split()
if filt:
    expids = [e for e in expids if filt in e]
for index, expid in enumerate(expids):
    print(f"{expid} {start + index % nproc}")
PY

wc -l < "${JOBS}"
xargs -P "${NPROC}" -L 1 bash scripts/run_one_mtl_baseline.sh < "${JOBS}"
echo "queue finished: ${JOBS}"
