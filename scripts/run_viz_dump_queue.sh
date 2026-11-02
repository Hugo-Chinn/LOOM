#!/usr/bin/env bash
chunk=$1; gpu=$2
cd /home/HugoChinn/LoopMT/UniRank
while read -r line; do
  [ -z "$line" ] && continue
  cfg=$(echo "$line" | cut -d'|' -f1); expid=$(echo "$line" | cut -d'|' -f2); ck=$(echo "$line" | cut -d'|' -f3)
  case "$cfg" in
    *onetrans*) prefix=onetrans ;;
    *mtl_baselines*) prefix=mtlbases ;;
    *) prefix=rankmixer ;;
  esac
  out="/home/HugoChinn/LoopMT/reports/repr_taobao/${prefix}__${expid}.npz"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
    .venv-unirank/bin/python scripts/dump_baseline_representations.py \
      --config "$cfg" --expid "$expid" --checkpoint-dir "$ck" --gpu 0 \
      --batches 3 --max-samples 16384 --out "$out" \
      > "/tmp/reprdump2_${prefix}_${expid}.out" 2>&1
  echo "done ${prefix}__${expid} (gpu $gpu)"
done < "$chunk"
