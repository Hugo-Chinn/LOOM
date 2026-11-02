#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
for spec in \
  'Taobao_Action taobao RankMixer_Taobao_Action_LocalSmoke 256' \
  'KuaiRand_Video_Action kuairand RankMixer_KuaiRand_Video_Action_LocalSmoke 256' \
  'MerRec_Action merrec RankMixer_MerRec_Action_LocalSmoke 128' \
  'QK_Video_Action qk RankMixer_QK_Video_Action_LocalSmoke 256'; do
  read -r dataset short expid batch <<<"$spec"
  "$PY" scripts/audit_preprocessed_dataset.py "data/preprocessed/$dataset" \
    --output "../reports/unirank_${short}_audit.json"
  echo "Validated audit for $dataset; real-batch smoke command:"
  echo "CUDA_VISIBLE_DEVICES=<free_gpu> $PY scripts/validate_preprocessed_rankmixer.py --expid $expid --output ../reports/unirank_${short}_real_batch_validation.json"
done
"$PY" -m pytest -q tests/test_environment_and_smoke.py
