#!/bin/bash
#SBATCH --job-name=dflash_qwen2_5_vl_7b_cauldron_online
#SBATCH --nodes=1
#SBATCH -p 36x2-a01r
#SBATCH -A coreai_mlperf_inference
#SBATCH -t 05:00:00
#SBATCH --signal=TERM@600

# Submit the standalone Cauldron pipeline in the vLLM container. Export and
# regeneration resume when OUTPUT_DIR persists across requeues.

set -uo pipefail
mkdir -p logs

# Cauldron source and reusable pool
export CAULDRON_DATASET_PATH="${CAULDRON_DATASET_PATH:-}"
export CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-}"
export CAULDRON_ALLOW_DOWNLOAD="${CAULDRON_ALLOW_DOWNLOAD:-}"
export PERC_SAMPLES="${PERC_SAMPLES:-0.1}"
export EXPORT_SEED="${EXPORT_SEED:-0}"
export MAX_SAMPLES="${MAX_SAMPLES-5000}"

# Model, training, and checkpointing
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-3e-4}"
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
export SAVE_BEST="${SAVE_BEST:-}"
export SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export MAX_ANCHORS="${MAX_ANCHORS:-3072}"
export NUM_LAYERS="${NUM_LAYERS:-5}"
export PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"
export LOSS_FN="${LOSS_FN:-ce}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"

# Regeneration and serving
export REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
export REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
export SKIP_REGEN="${SKIP_REGEN:-}"
export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\": 1003520\}}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4\}}"
export REGEN_GPUS="${REGEN_GPUS:-0,1,2,3}"
export REGEN_DP="${REGEN_DP:-1}"
export REGEN_TP="${REGEN_TP:-4}"
export EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
export EXTRACT_DP_SIZE="${EXTRACT_DP_SIZE:-2}"
export EXTRACT_TP="${EXTRACT_TP:-1}"
export TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
export SERVER_PORT="${SERVER_PORT:-9090}"

# Container and persistent paths
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/coreai_mlperf_inference/jcalderon/containers/vllm_0.28.0.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw/coreai_mlperf_inference/jcalderon/:/workspace}"
WORK_DIR="${WORK_DIR:-/workspace/sandbox/dflash_training/speculators}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/sandbox/dflash_training/output/dflash_qwen2_5_vl_7b_cauldron}"
export HF_HOME="${HF_HOME:-/workspace/.cache}"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_DISABLE_COMPILE_CACHE=1

COMMENTS="${COMMENTS:-}"
echo "Submitting: subsets=${CAULDRON_SUBSETS:-all} perc_samples=$PERC_SAMPLES max_samples=${MAX_SAMPLES:-all}"

srun --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    /bin/bash -c "
    set -uo pipefail
    cd ${WORK_DIR} || { echo 'WORK_DIR ${WORK_DIR} not found in container' >&2; exit 1; }
    echo '--- provenance ---'
    sha1sum slurm/training_script_cauldron.sh examples/train/dflash_qwen2_5_vl_7b_cauldron_online.sh scripts/export_cauldron.py
    python3 -c 'import sys; print(sys.executable, sys.version.split()[0])'
    echo '------------------'
    unset VLLM_PORT VLLM_DP_SIZE
    python3 -m pip install 'datasets>=4.0.0,<=5.0.1' || {
        echo 'datasets install failed' >&2
        exit 1
    }
    python3 -m pip install --no-deps -e ./hs_connectors -e . || {
        echo 'Editable install failed; needs git plus network for build deps' >&2
        exit 1
    }
    python3 -c 'import datasets, hs_connectors, speculators, speculators.train.data' || {
        echo 'Preflight import check failed' >&2
        exit 1
    }
    bash examples/train/dflash_qwen2_5_vl_7b_cauldron_online.sh
" > "logs/dflash_cauldron_training_${COMMENTS}_${SLURM_JOB_ID}.log" 2>&1 &
wait
