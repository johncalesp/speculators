#!/bin/bash
# Runs inside the container. Use its Python, or explicitly set PYTHON.
set -euo pipefail
export HF_HOME="${HF_HOME:-/workspace/.cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ENABLE_SCALE_OUT_ENDPOINTS=1
export PYTHONPATH="$PWD/src:$PWD/hs_connectors/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python}" slurm/cauldron_pipeline.py
