#!/usr/bin/env bash
# Run only the missing seeds; wait non-preemptively for GPUs 0-3.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_safe_routing_multiseed"
OUT="$ROOT/runs_local/rankmixer_safe_routing_multiseed"
mkdir -p "$OUT"
"$PY" scripts/build_rankmixer_safe_routing_multiseed.py

EXPS=(
  RankMixer_Kuai_AdapterOnly_S20262029
  RankMixer_Kuai_AdapterResidualBypass_S20262029
  RankMixer_Kuai_AdapterOnly_S20262030
  RankMixer_Kuai_AdapterResidualBypass_S20262030
)
GPUS=(0 1 2 3)

wait_for_gpu() {
  local gpu="$1" used
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ ]] && (( used < 2048 )); then
      return
    fi
    sleep 30
  done
}

pids=()
for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"; gpu="${GPUS[$i]}"
  (
    wait_for_gpu "$gpu"
    printf '%s START gpu=%s %s\n' "$(date '+%F %T')" "$gpu" "$exp" | tee -a "$OUT/queue.log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py \
      --config "$CONFIG" --expid "$exp" --gpu 0 \
      > "$OUT/$exp.stdout.log" 2>&1
    printf '%s DONE gpu=%s %s\n' "$(date '+%F %T')" "$gpu" "$exp" | tee -a "$OUT/queue.log"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
"$PY" scripts/summarize_rankmixer_safe_routing_multiseed.py \
  > "$OUT/summary.json"
exit "$failed"
