#!/bin/bash
#SBATCH --job-name=dflash_qwen2_5_vl_7b_visionarena_online
#SBATCH --nodes=1
#SBATCH -p 36x2-a01r
#SBATCH -A coreai_mlperf_inference
#SBATCH -t 05:00:00
# Deliver SIGTERM 10 minutes before the walltime instead of at the very end.
# train.py catches it and writes an `interrupted` checkpoint, but it allows
# itself up to 120s to do so -- with no --signal, Slurm sends SIGTERM only at
# expiry and SIGKILLs after KillWait (often 30s), truncating that write.
#SBATCH --signal=TERM@600

# Submits examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh inside the
# vLLM container. Every knob is an environment variable with a default, so
# scaling up needs no edit to either file:
#
#   EXPORT_LIMIT=200000 MAX_SAMPLES=200000 EPOCHS=2 sbatch slurm/training_script.sh
#
# Note the 5h walltime above. Regenerating on-policy responses is the dominant
# cost of a large run, so raise -t before raising EXPORT_LIMIT. The export, the
# regeneration and the training all resume, so a job that hits the limit can be
# resubmitted and will pick up where it stopped as long as OUTPUT_DIR persists.

set -uo pipefail
mkdir -p logs

# ============ Pipeline configuration ============
# srun propagates the submitting environment into the container, so exporting
# here is enough for the training script to pick these up.

# Data scale. EXPORT_LIMIT is a target size for the prompts file, not a per-run
# quota, so resubmitting tops it up instead of appending another batch.
export EXPORT_LIMIT="${EXPORT_LIMIT:-5000}"
export MAX_SAMPLES="${MAX_SAMPLES:-5000}"
export MAX_TURNS="${MAX_TURNS:-2}"
export EXPORT_LANGUAGE="${EXPORT_LANGUAGE:-English}"   # e.g. English; empty keeps all

# Training
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-3e-4}"

# Checkpointing. 1 saves once per epoch; below 1 saves within an epoch
# (0.25 = four times per epoch), worth setting for large single-epoch runs that
# would otherwise lose hours of work to a walltime. Set SAVE_BEST non-empty to
# prune everything but the best-validation-loss checkpoint; unset keeps all
# epochs, each carrying model weights plus optimizer state.
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
export SAVE_BEST="${SAVE_BEST:-}"

# Regeneration throughput
export REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
export REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"

# DFlash recipe (RFC #979 best practices)
export SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
export BLOCK_SIZE="${BLOCK_SIZE:-16}"
export MAX_ANCHORS="${MAX_ANCHORS:-3072}"
export NUM_LAYERS="${NUM_LAYERS:-5}"
export PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"
export LOSS_FN="${LOSS_FN:-ce}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
# Qwen2.5-VL-7B has 28 text layers. Must match vLLM's aux hidden state layers.
export TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"

# Image token budget. Keep both servers on the same value so responses are
# generated at the resolution the drafter is trained on.
export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\": 1003520\}}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4\}}"

# GPU layout. Defaults assume 4 visible GPUs; the training script fails fast if
# the allocation is smaller. Align these with whatever this partition grants.
export REGEN_GPUS="${REGEN_GPUS:-0,1,2,3}"
export REGEN_TP="${REGEN_TP:-4}"
export VLLM_GPUS="${VLLM_GPUS:-0,1}"
export TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
export VLLM_DP_SIZE="${VLLM_DP_SIZE:-2}"

# Named VLLM_HTTP_PORT, not VLLM_PORT: vLLM reads VLLM_PORT from its own
# environment, so exporting that name would reconfigure the server itself.
export VLLM_HTTP_PORT="${VLLM_HTTP_PORT:-9090}"

# Container and paths
CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/coreai_mlperf_inference/jcalderon/containers/vllm_0.27.1.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw/coreai_mlperf_inference/jcalderon/:/workspace}"
WORK_DIR="${WORK_DIR:-/workspace/sandbox/dflash_training/speculators}"

export MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
# Keep run artifacts on the mounted filesystem so --resume survives requeues.
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/sandbox/dflash_training/output/dflash_qwen2_5_vl_7b_visionarena}"

export HF_HOME="${HF_HOME:-/workspace/.cache}"
export VLLM_DISABLE_COMPILE_CACHE=1
# HF_TOKEN is read from the submitting environment rather than written here.
# With HF_HOME on the mounted filesystem, `huggingface-cli login` once is
# enough and no token needs to live in this file at all.
export HF_TOKEN="${HF_TOKEN:-}"

COMMENTS="${COMMENTS:-}"
# =================================================

echo "Submitting: export_limit=${EXPORT_LIMIT} max_samples=${MAX_SAMPLES} epochs=${EPOCHS}"

srun --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    /bin/bash -c "
    set -uo pipefail
    cd ${WORK_DIR} || { echo 'WORK_DIR ${WORK_DIR} not found in container' >&2; exit 1; }
    unset VLLM_PORT
    # Invoked via bash: the example scripts are not marked executable.
    bash examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh
" > "logs/dflash_training_${COMMENTS}_${SLURM_JOB_ID}.log" 2>&1 &
wait
