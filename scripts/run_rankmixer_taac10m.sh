#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_LIST="${GPU_LIST:-2,3,4,5}"
NPROC="${NPROC:-4}"
PORT="${MASTER_PORT:-29620}"
EXPERIMENT_ID="${EXPERIMENT_ID:-RankMixer_TencentGR_10M_Action_ReproV001}"

test -f "$ROOT/data/TAAC2025/TencentGR_10M_Action/block_manifest.json" || {
  echo "TAAC blocked dataset is not ready" >&2; exit 1;
}
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/runs_local/taac_rankmixer_repro/torchinductor"
export TRITON_CACHE_DIR="$ROOT/runs_local/taac_rankmixer_repro/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "$ROOT/.venv-unirank/bin/python" -m torch.distributed.run \
  --standalone --master_port="$PORT" \
  --nproc_per_node="$NPROC" run_expid.py --config ./config_local \
  --expid "$EXPERIMENT_ID" --gpu "$(seq -s, 0 $((NPROC - 1)))"
