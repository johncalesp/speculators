#!/bin/bash
#SBATCH --job-name=dflash2_qwen2_5_vl_7b_cauldron_online
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -p 36x2-a01r
#SBATCH -A coreai_mlperf_inference
#SBATCH -t 05:00:00

# Submit from the repo root: sbatch slurm/training_script_cauldron.sh
# Exit 75 requests one dependent continuation. Ordinary failures stop the chain.
set -euo pipefail

CONTAINER_IMAGE="${CONTAINER_IMAGE:-/lustre/fsw/coreai_mlperf_inference/jcalderon/containers/vllm_0.28.0.sqsh}"
CONTAINER_MOUNTS="${CONTAINER_MOUNTS:-/lustre/fsw/coreai_mlperf_inference/jcalderon/:/workspace}"
WORK_DIR="${WORK_DIR:-/workspace/sandbox/dflash2_training/speculators}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/sandbox/dflash2_training/output/dflash2_qwen2_5_vl_7b_cauldron}"
export HF_HOME="${HF_HOME:-/workspace/.cache}"
export HF_TOKEN="${HF_TOKEN:-}"
export VLLM_DISABLE_COMPILE_CACHE=1
export EPOCHS="${EPOCHS:-5}"
export CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-all}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
# Include container startup; reserve 30 minutes for checkpointing and cleanup.
if (( ${WORK_SECONDS:-16200} < 1 || ${WORK_SECONDS:-16200} > 16200 || ${SAVE_GRACE_SECONDS:-1200} > 1200 )); then
    echo "WORK_SECONDS must be 1..16200 and SAVE_GRACE_SECONDS <= 1200" >&2
    exit 2
fi
export CAULDRON_DEADLINE=$(( $(date +%s) + ${WORK_SECONDS:-16200} ))
export SLURM_EXPORT_ENV=ALL
export PIPELINE_SBATCH_SCRIPT="${PIPELINE_SBATCH_SCRIPT:-${SLURM_SUBMIT_DIR:?}/slurm/training_script_cauldron.sh}"
test -f "$PIPELINE_SBATCH_SCRIPT"

set +e
srun --ntasks=1 --kill-on-bad-exit=1 \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-workdir="$WORK_DIR" \
    bash slurm/run_cauldron.sh
rc=$?
set -e
if (( rc == 75 )); then
    if [[ "${AUTO_CONTINUE:-1}" == 1 ]]; then
        next_job=$(sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
            --partition="$SLURM_JOB_PARTITION" --account="$SLURM_JOB_ACCOUNT" \
            --export=ALL "$PIPELINE_SBATCH_SCRIPT")
        echo "Saved progress. Continuation job: $next_job"
    else
        echo "Saved progress. AUTO_CONTINUE=0; submit this script again to resume."
    fi
    exit 0
fi
exit "$rc"
