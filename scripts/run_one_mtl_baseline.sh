#!/usr/bin/env bash
# Run one multi-task baseline arm on a single GPU.
# Usage: run_one_mtl_baseline.sh <expid> <local_gpu_index>
set -uo pipefail
cd "$(dirname "$0")/.."

EXPID="$1"
GPU="${2:-0}"
PY="${PY:-./.venv-unirank/bin/python}"
CONFIG="./config_mtl_baselines_four_seed/"
STAMP_DIR="./logs/mtl_baselines_four_seed"

mkdir -p "${STAMP_DIR}" ./runs_local
LOG="${STAMP_DIR}/${EXPID}.log"

if [[ -f "${STAMP_DIR}/${EXPID}.done" ]]; then
    echo "skip ${EXPID}"
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
if ! "${PY}" run_expid.py --config "${CONFIG}" --expid "${EXPID}" --gpu 0 >"${LOG}" 2>&1; then
    # One retry: transient NCCL / dataloader start-up failures are common.
    if ! "${PY}" run_expid.py --config "${CONFIG}" --expid "${EXPID}" --gpu 0 >"${LOG}" 2>&1; then
        echo "FAIL ${EXPID}"
        touch "${STAMP_DIR}/${EXPID}.fail"
        exit 1
    fi
fi
touch "${STAMP_DIR}/${EXPID}.done"
echo "done ${EXPID} (gpu ${GPU})"
