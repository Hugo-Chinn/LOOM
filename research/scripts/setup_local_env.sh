#!/usr/bin/env bash
# Recreate the isolated UniRank environment on this machine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv-unirank"
SOURCE_ENV="${UNIRANK_TORCH_SOURCE_ENV:-/apdcephfs_qy3/share_305164797/haibinlv/gen_rec/envs/sid_v2_512x3}"
SOURCE_SITE="$SOURCE_ENV/lib64/python3.11/site-packages"

if [[ ! -d "$VENV" ]]; then
  uv venv "$VENV" --python 3.11
fi
"$VENV/bin/python" -m ensurepip --upgrade

# The host's validated CUDA build is reused read-only; every other dependency
# is installed into this venv. This avoids modifying or importing the host
# environment as a whole while avoiding a second multi-GB CUDA download.
VENV_SITE="$VENV/lib/python3.11/site-packages"
for entry in torch torch-2.6.0+cu124.dist-info torchgen functorch nvidia \
             typing_extensions.py typing_extensions-4.16.0.dist-info \
             triton triton-3.2.0.dist-info sympy sympy-1.13.1.dist-info; do
  [[ -e "$VENV_SITE/$entry" ]] || ln -s "$SOURCE_SITE/$entry" "$VENV_SITE/$entry"
done

"$VENV/bin/python" -m pip install --no-deps \
  keras_preprocessing pandas==2.3.3 PyYAML scikit-learn numpy h5py \
  heavyball==3.2.0 tqdm huggingface_hub pyarrow 'polars<=1.0.0' datasets pytest
"$VENV/bin/python" -m pip install \
  filelock 'fsspec[http]<=2026.6.0,>=2023.1.0' jinja2 networkx six joblib \
  narwhals scipy threadpoolctl iniconfig packaging pluggy pygments \
  python-dateutil pytz tzdata click hf-xet httpx opt-einsum dill multiprocess \
  requests xxhash aiohttp anyio certifi httpcore idna h11 cloudpickle \
  charset_normalizer urllib3 'mpmath<1.4,>=1.1.0'
"$VENV/bin/python" -m pip check
"$VENV/bin/python" -m pip freeze | sort > "$ROOT/environment.lock.txt"
"$VENV/bin/python" "$ROOT/scripts/check_environment.py"
