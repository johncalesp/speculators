#!/bin/bash
#SBATCH --job-name=dflash_qwen2_5_vl_7b_cauldron_online
#SBATCH --nodes=1
#SBATCH -p 36x2-a01r
#SBATCH -A coreai_mlperf_inference
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

# Submit the standalone Cauldron pipeline in the vLLM container. Export and
# regeneration resume when OUTPUT_DIR persists across requeues. Slurm signals
# this batch shell ten minutes before walltime; it stops the active step cleanly
# and requeues the same job until training reaches EPOCHS.

set -uo pipefail
mkdir -p logs

SRUN_PID=""
REQUEUE_IN_PROGRESS=0

requeue_job() {
    if (( REQUEUE_IN_PROGRESS )); then
        return
    fi
    REQUEUE_IN_PROGRESS=1
    trap '' USR1
    echo "Walltime signal received; stopping job step before requeue."
    if [[ -n "$SRUN_PID" ]] && kill -0 "$SRUN_PID" 2>/dev/null; then
        # srun forwards TERM to torchrun. The trainer has up to 120 seconds to
        # save its interrupted checkpoint, while the latest numeric fractional
        # checkpoint remains the automatic resume point.
        kill -TERM "$SRUN_PID" 2>/dev/null || true
        wait "$SRUN_PID" 2>/dev/null || true
    fi
    SRUN_PID=""
    echo "Requeueing Slurm job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0})."
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        echo "scontrol requeue failed; the job will not restart automatically." >&2
        exit 1
    fi
    exit 0
}
trap requeue_job USR1

# Cauldron source and reusable pool
export CAULDRON_DATASET_PATH="${CAULDRON_DATASET_PATH:-}"
export CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-}"
export CAULDRON_ALLOW_DOWNLOAD="${CAULDRON_ALLOW_DOWNLOAD:-}"
export PERC_SAMPLES="${PERC_SAMPLES:-0.1}"
export EXPORT_SEED="${EXPORT_SEED:-0}"
export MAX_SAMPLES="${MAX_SAMPLES-5000}"
export CAULDRON_PROFILE="${CAULDRON_PROFILE:-llava_wild}"
export CAULDRON_TRAIN_SUBSETS="${CAULDRON_TRAIN_SUBSETS:-}"
export VAL_FRACTION="${VAL_FRACTION:-0.1}"
export MAX_QUESTIONS_PER_IMAGE="${MAX_QUESTIONS_PER_IMAGE:-1}"

# Model, training, and checkpointing
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export EPOCHS="${EPOCHS:-5}"
export LR="${LR:-3e-4}"
# A numeric checkpoint is overwritten every 10% of an epoch. This permits
# exact mid-epoch resume even when one epoch exceeds the five-hour allocation.
export CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-0.1}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"
export SAVE_BEST="${SAVE_BEST:-}"
export SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
export BLOCK_SIZE="${BLOCK_SIZE:-5}"
export MAX_ANCHORS="${MAX_ANCHORS:-3072}"
export NUM_LAYERS="${NUM_LAYERS:-5}"
export PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"
export LOSS_FN="${LOSS_FN:-ce}"
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-152064}"
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
echo "Submitting: subsets=${CAULDRON_SUBSETS:-all} profile=$CAULDRON_PROFILE perc_samples=$PERC_SAMPLES max_samples=${MAX_SAMPLES:-all}"
echo "Slurm restart=${SLURM_RESTART_COUNT:-0} checkpoint_freq=$CHECKPOINT_FREQ"

if [[ -n "$SAVE_BEST" ]] && awk -v value="$CHECKPOINT_FREQ" \
    'BEGIN { exit !(value < 1) }'; then
    echo "SAVE_BEST disables fractional checkpoints and is unsafe with five-hour requeues." >&2
    echo "Leave SAVE_BEST empty so CHECKPOINT_FREQ=$CHECKPOINT_FREQ can resume mid-epoch." >&2
    exit 1
fi

LOG_FILE="logs/dflash_cauldron_training_${COMMENTS}_${SLURM_JOB_ID}.log"
srun --container-image="${CONTAINER_IMAGE}" --container-mounts="${CONTAINER_MOUNTS}" \
    /bin/bash -c "
    set -uo pipefail
    cd ${WORK_DIR} || { echo 'WORK_DIR ${WORK_DIR} not found in container' >&2; exit 1; }
    echo '--- provenance ---'
    sha1sum slurm/training_script_cauldron.sh examples/train/dflash_qwen2_5_vl_7b_cauldron_online.sh scripts/export_cauldron.py scripts/select_cauldron_data.py
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
" >> "$LOG_FILE" 2>&1 &
SRUN_PID=$!
wait "$SRUN_PID"
STATUS=$?
SRUN_PID=""
exit "$STATUS"
