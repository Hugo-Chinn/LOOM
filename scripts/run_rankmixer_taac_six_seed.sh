#!/usr/bin/env bash
# Six-seed TencentGR confirmation: Loop_GatedDeep_Pre2 / Loop_Deep vs Sequential.
# Completed logs are skipped, so the script is safe to restart.
#
# Execution namespace: the default Codex sandbox does not expose /dev/nvidia*, so
# this script must run in the approved non-sandbox context (`nvidia-smi` then
# reports 8x H20). Launch it detached, for example
#   setsid nohup bash scripts/run_rankmixer_taac_six_seed.sh \
#     > runs_local/rankmixer_taac_six_seed/launcher.log 2>&1 < /dev/null &
# without setsid the children are killed when the launching session is reaped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_taac_six_seed"
RUN_DIR="$ROOT/runs_local/rankmixer_taac_six_seed"
mkdir -p "$RUN_DIR"
"$PY" scripts/build_rankmixer_taac_six_seed.py

SEEDS=(20262031 20262032 20262033)
ARMS=(Sequential Loop_Deep Loop_GatedDeep_Pre2)
WAVE_SIZE=6

run_wave() {
  local jobs=("$@") pids=() gpu=0 expid log
  for expid in "${jobs[@]}"; do
    log="$RUN_DIR/${expid}.stdout.log"
    if [[ -f "$log" ]] && grep -q 'Training finished.' "$log" && grep -q 'Test evaluation' "$log"; then
      echo "SKIP completed $expid"
      continue
    fi
    echo "START gpu=$gpu $expid"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py --config "$CONFIG" \
      --expid "$expid" --gpu 0 > "$log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

jobs=()
for seed in "${SEEDS[@]}"; do
  for arm in "${ARMS[@]}"; do jobs+=("RankMixer_TAAC_${arm}_S${seed}"); done
done

for ((i = 0; i < ${#jobs[@]}; i += WAVE_SIZE)); do
  run_wave "${jobs[@]:i:WAVE_SIZE}"
  echo "WAVE DONE through $((i + WAVE_SIZE)) of ${#jobs[@]}"
done
echo "ALL DONE"
