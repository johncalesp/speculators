#!/bin/bash
# Runs inside the container. Use its Python, or explicitly set PYTHON.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHON="${PYTHON:-python3}"
export HF_HOME="${HF_HOME:-/workspace/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HF_HOME/pip}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1
export PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
# Parallelism is set explicitly by the pipeline; discard legacy launcher state.
unset VLLM_DP_SIZE

echo "Container startup source hashes:"
sha256sum slurm/training_script_cauldron.sh slurm/run_cauldron.sh \
    slurm/cauldron_pipeline.py slurm/cauldron_data.py \
    slurm/cauldron_state.py slurm/train_cauldron.py
"$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])'

# Each continuation starts a fresh container. Reinstall local packages and let
# pip reuse satisfied dependencies and its persistent download/build cache.
"$PYTHON" -m pip install 'datasets>=4.0.0,<=5.0.1' || {
    echo "datasets install failed" >&2
    exit 1
}
"$PYTHON" -m pip install --no-deps -e ./hs_connectors -e . || {
    echo "Editable install failed; needs git plus network for build dependencies" >&2
    exit 1
}
"$PYTHON" -c 'import datasets, hs_connectors, speculators, speculators.train.data' || {
    echo "Preflight import check failed" >&2
    exit 1
}

# Forward the pipeline's exit code (including 75) directly to the Slurm wrapper.
exec "$PYTHON" slurm/cauldron_pipeline.py
