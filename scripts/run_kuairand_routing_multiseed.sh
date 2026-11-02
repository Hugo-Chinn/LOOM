#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
OUT="$ROOT/runs_local/kuairand_routing_multiseed"
mkdir -p "$OUT"
"$PY" scripts/build_kuairand_routing_multiseed_config.py

EXPS=(
  Kuai_Bypass_S20262028 Kuai_Full_S20262028
  Kuai_Bypass_S20262029 Kuai_Full_S20262029
  Kuai_Bypass_S20262030 Kuai_Full_S20262030
)
GPUS=(2 3 4 5 6 7)
pids=()
for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"; gpu="${GPUS[$i]}"
  (CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py \
    --config ./config_kuairand_routing_multiseed --expid "$exp" --gpu 0 \
    > "$OUT/$exp.stdout.log" 2>&1) &
  pids+=("$!")
  echo "$! $gpu $exp"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
