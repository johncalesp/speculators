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

# Which dataset to train on: visionarena or nemotron. The rest of the pipeline
# is identical either way; only step 1 differs.
export DATASET="${DATASET:-visionarena}"
# Directory holding the already-downloaded dataset. Empty reads the HuggingFace
# cache at HF_HOME below.
export DATASET_PATH="${DATASET_PATH:-}"
# Set non-empty to stream visionarena from the Hub instead of reading a local
# copy. Off by default: compute nodes often have no route to the Hub, and
# pulling 84 GB of shards inside the job wastes the allocation.
export EXPORT_ALLOW_DOWNLOAD="${EXPORT_ALLOW_DOWNLOAD:-}"

# Data scale. EXPORT_LIMIT is a target size for the prompts file, not a per-run
# quota, so resubmitting tops it up instead of appending another batch.
# EXPORT_LIMIT, MAX_TURNS and EXPORT_LANGUAGE apply to visionarena only.
export EXPORT_LIMIT="${EXPORT_LIMIT:-5000}"
export MAX_TURNS="${MAX_TURNS:-2}"
# ${VAR-default}, not ${VAR:-default}: both of these have to be settable to the
# empty string to mean "no cap" and "every language". With :- an explicit
# MAX_SAMPLES= would silently fall back to 5000 and cap a full-dataset run.
export MAX_SAMPLES="${MAX_SAMPLES-5000}"
export EXPORT_LANGUAGE="${EXPORT_LANGUAGE-English}"    # e.g. English; empty keeps all

# --- nemotron only ---
# Comma-separated partitions, e.g. vqa_1 or ocr_1,vqa_9. Empty uses every
# partition whose images are already on disk. Run
#   python3 scripts/export_nemotron_vlm.py --list-partitions
# to see the inventory.
export NEMOTRON_PARTITIONS="${NEMOTRON_PARTITIONS:-}"
# Size knob for nemotron, in place of MAX_SAMPLES: 1 = 100%, 0.05 = 5%.
export EXPORT_FRACTION="${EXPORT_FRACTION:-0.01}"
# Shared by the image download and the export so they agree on which rows are
# in play. Changing it selects a different subset of the same size and orphans
# the images already fetched.
export EXPORT_SEED="${EXPORT_SEED:-0}"
# Partitions whose images the repo does not ship but which can be fetched from
# OpenImages (vqa_1, vqa_2, vqa_3, captioning_1, captioning_2) are downloaded in
# step 0, at EXPORT_FRACTION -- so only the images the export will use are
# fetched. Set non-empty to require them present already and fail instead.
export NEMOTRON_SKIP_DOWNLOAD="${NEMOTRON_SKIP_DOWNLOAD:-}"
export NEMOTRON_DOWNLOAD_CONCURRENCY="${NEMOTRON_DOWNLOAD_CONCURRENCY:-64}"

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
# Defaults to OUTPUT_DIR/checkpoints. Point it somewhere new when scaling a run
# up: the trainer resumes from whatever is here, and once the saved epochs reach
# EPOCHS it would train nothing at all. Keeping OUTPUT_DIR unchanged still lets
# the export and the regeneration reuse what they already produced.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-}"

# Regeneration throughput
export REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
export REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
# Set non-empty to skip regeneration and train on the conversations already on
# disk. Worth using to continue an interrupted training run: topping up a few
# failed conversations changes the conversation count, which invalidates the
# prepared dataset and costs a full re-render of every row to recover them.
export SKIP_REGEN="${SKIP_REGEN:-}"

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
# REGEN_DP * REGEN_TP must equal the REGEN_GPUS count, and REGEN_TP is capped by
# the target's attention-head count (28 for Qwen2.5-VL-7B: 1, 2 or 4, not 8).
# Beyond that cap, scale regeneration with replicas rather than wider tensors.
export REGEN_DP="${REGEN_DP:-1}"
export REGEN_TP="${REGEN_TP:-4}"
export EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
export TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
export NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
# EXTRACT_DP_SIZE * EXTRACT_TP must equal the EXTRACT_GPUS count. Switch to
# EXTRACT_DP_SIZE=1 EXTRACT_TP=2 if vLLM refuses data parallelism for this model.
export EXTRACT_DP_SIZE="${EXTRACT_DP_SIZE:-2}"
export EXTRACT_TP="${EXTRACT_TP:-1}"

# None of the names above may begin with VLLM_. vLLM owns that namespace and
# applies those settings from the environment, so exporting one silently
# reconfigures every server this pipeline starts. EXTRACT_DP_SIZE was VLLM_DP_SIZE
# and switched on offline data parallelism for the regeneration server in step 2,
# which never passes --data-parallel-size at all; vLLM refused to start with
# "Offline data parallel mode is not supported/useful for dense models".
export SERVER_PORT="${SERVER_PORT:-9090}"

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
    # Which copies of the two scripts actually ran, and which interpreter. The
    # work dir is a separate checkout from wherever this file was edited, so a
    # stale copy otherwise shows up as a baffling downstream error.
    echo '--- provenance ---'
    sha1sum slurm/training_script.sh examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh
    python3 -c 'import sys; print(sys.executable, sys.version.split()[0])'
    echo '------------------'
    # Drop any inherited vLLM settings that would override the flags the
    # pipeline passes explicitly, in case the submitting shell has them set.
    unset VLLM_PORT VLLM_DP_SIZE
    # The container ships vLLM's whole stack but not 'datasets', and neither
    # 'speculators' nor 'hs_connectors' is installed at all. --no-deps on the two
    # editable installs keeps pip from touching the container's pinned
    # torch/transformers/vllm. speculators has to be a real install rather than
    # just PYTHONPATH: config.py reads its own version via importlib.metadata
    # while the class body executes, so without dist metadata the import raises
    # PackageNotFoundError. hs_connectors is an unguarded top-level import in
    # speculators.train.data, so it is required too.
    # 'python3 -m pip' rather than bare 'pip' so the install always targets the
    # interpreter the pipeline runs under, not whichever pip happens to be first
    # on PATH.
    python3 -m pip install 'datasets>=4.0.0,<=5.0.1' || { echo 'datasets install failed' >&2; exit 1; }
    python3 -m pip install --no-deps -e ./hs_connectors -e . || {
        echo 'Editable install failed; needs git plus network for build deps' >&2
        exit 1
    }
    # Fail in seconds rather than after the multi-hour export and regeneration.
    python3 -c 'import datasets, hs_connectors, speculators, speculators.train.data' || {
        echo 'Preflight import check failed' >&2
        exit 1
    }
    # Invoked via bash: the example scripts are not marked executable.
    bash examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh
" > "logs/dflash_training_${COMMENTS}_${SLURM_JOB_ID}.log" 2>&1 &
wait
