#!/usr/bin/env bash
# Train-with-checkpoint workers for the Taobao representation study.
chunk=$1; gpu=$2
cd /home/HugoChinn/LoopMT/UniRank
while read -r line; do
  [ -z "$line" ] && continue
  cfg=$(echo "$line" | cut -d'|' -f1); expid=$(echo "$line" | cut -d'|' -f2)
  # coexist with the TAAC sweep: start only if this GPU has room
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    [ "${used:-0}" -lt 40000 ] && break
    sleep 60
  done
  CUDA_VISIBLE_DEVICES=$gpu TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_viztb_$expid \
    .venv-unirank/bin/python run_expid.py --config "$cfg" --expid "$expid" --gpu 0 \
    > "/tmp/viztb_$expid.out" 2>&1
  echo "done $expid (gpu $gpu)"
done < "$chunk"
