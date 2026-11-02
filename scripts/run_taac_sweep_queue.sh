#!/usr/bin/env bash
# Sequential worker: one TAAC run per GPU at a time.
chunk=$1; gpu=$2
cd /home/HugoChinn/LoopMT/UniRank
while read -r expid; do
  [ -z "$expid" ] && continue
  # wait until this GPU is (nearly) free before starting
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    [ "${used:-0}" -lt 8000 ] && break
    sleep 120
  done
  CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_taacsw_$expid \
    .venv-unirank/bin/python run_expid.py --config ./config_rankmixer_taac_sweep_20261025 --expid "$expid" --gpu 0 \
    > "/tmp/taacsw_$expid.out" 2>&1
  echo "done $expid (gpu $gpu)"
done < "$chunk"
