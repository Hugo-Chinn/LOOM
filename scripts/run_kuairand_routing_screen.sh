#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
OUT="$ROOT/runs_local/kuairand_routing_screen"
mkdir -p "$OUT"
"$PY" scripts/build_kuairand_routing_screen_config.py

# The current control has already been completed under the identical protocol,
# so launch only the four new mechanism-isolating arms to obtain results sooner.
EXPS=(Kuai_Bypass Kuai_Dynamic Kuai_Dynamic_Bypass Kuai_Dynamic_Bypass_Adapter)
GPUS=(2 3 4 5)
pids=()
for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"; gpu="${GPUS[$i]}"
  (CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py \
    --config ./config_kuairand_routing_screen --expid "$exp" --gpu 0 \
    > "$OUT/$exp.stdout.log" 2>&1) &
  pids+=("$!")
  echo "$! $gpu $exp"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
