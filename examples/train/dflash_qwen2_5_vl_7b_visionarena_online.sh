#!/bin/bash
# Online DFlash Training Script -- Vision-Language Target Model
#
# Trains a DFlash drafter for Qwen2.5-VL-7B-Instruct on real user vision prompts
# from lmarena-ai/VisionArena-Chat, with responses regenerated on-policy by the
# target model itself.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh
#
# For a detailed walkthrough of the text-only pipeline, see
# https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/
# and for the DFlash recipe rationale, https://github.com/vllm-project/speculators/issues/979
#
# ============================================================================
# Why this script has more steps than the text-only examples
# ============================================================================
#
# 1. VisionArena-Chat's assistant turns come from arena models (GPT-4o, Claude,
#    and others), not from Qwen2.5-VL. Training on them would teach the drafter
#    to predict another model's tokens. Steps 1-3 therefore keep only the user
#    prompts and their images and regenerate the responses with the target
#    model, which is what makes the data on-policy.
#
#    scripts/response_regeneration/ cannot do this: it emits speculator-format
#    input_ids/loss_mask rows, and prepare_data.py passes those through without
#    the `messages` column that multimodal hidden-state extraction needs, so the
#    images would be dropped and the image placeholder tokens embedded as plain
#    text. regenerate_vlm_responses.py emits conversations instead.
#
# 2. Regeneration and hidden-state extraction want different servers. The
#    hidden-states server writes a safetensors file for every request it serves,
#    so running multi-token regeneration against it would fill the disk. Step 2
#    launches a plain server for regeneration; step 5 launches the
#    hidden-states server for rendering and training.
#
# 3. Images are passed to vLLM as file:// URLs, so every server that touches
#    them needs --allowed-local-media-path. prepare_data.py deliberately
#    rejects inline and base64 images so preprocessed datasets never copy them.
#
# 4. prepare_data.py needs --render-endpoint, so it runs *after* a server is up
#    rather than as step 1. (The text-only examples in this directory still run
#    it first and predate that requirement.)
#
# ============================================================================
# Scaling up
# ============================================================================
#
# MAX_SAMPLES below is deliberately a smoke-test value: enough to verify the
# pipeline end to end and confirm the drafter is learning, not enough for a
# good model. Every setting reads from the environment, so scaling up needs no
# edit to this file -- raise EXPORT_LIMIT and MAX_SAMPLES together and lower
# EPOCHS:
#
#   EXPORT_LIMIT=200000 MAX_SAMPLES=200000 EPOCHS=2 \
#     bash examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh
#
# (the dataset has ~199k conversations total)
#
# Budget for it: regenerating 200k multi-turn responses is the dominant cost,
# far more than training. Step 1 expects the dataset to be downloaded already
# and reads it from disk, so raising EXPORT_LIMIT costs no bandwidth -- but the
# images it materializes do cost disk, roughly 0.4GB per 1000 conversations.
# Both the export and the regeneration take --resume, so a full-scale run can
# be stopped and restarted.

set -euo pipefail

# vLLM reads these from the environment and they override the flags this script
# passes, so a value left over in the calling shell silently reconfigures the
# servers below. VLLM_DP_SIZE is the dangerous one: it enables offline data
# parallelism, which vLLM then rejects for a dense model with an error that
# names neither the variable nor the environment. Cleared here rather than only
# in slurm/training_script.sh, so running this script directly is just as safe.
unset VLLM_PORT VLLM_DP_SIZE

# ============ Configuration ============
# Every value below can be overridden from the environment, so a scheduler
# script (see slurm/training_script.sh) can set them without editing this file:
#   EXPORT_LIMIT=200000 MAX_SAMPLES=200000 EPOCHS=2 bash examples/train/...sh
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen2_5_vl_7b_visionarena}"
# No variable this script reads may be named VLLM_*. vLLM claims that whole
# namespace for its own settings and applies them from the environment, so such
# a name silently reconfigures the server: VLLM_PORT overrides --port, and
# VLLM_DP_SIZE turns on offline data parallelism, which vLLM then rejects for a
# dense model with "Offline data parallel mode is not supported/useful". vLLM
# warns about VLLM_* names it does not recognize but says nothing about the ones
# it does, so a collision looks like an unrelated config error.
SERVER_PORT="${SERVER_PORT:-8000}"

# Which source dataset step 1 exports from. Steps 2-7 are identical either way,
# since both exports emit the same prompt-only conversations.
#   visionarena - lmarena-ai/VisionArena-Chat, real multi-turn user chat prompts
#   nemotron    - nvidia/Llama-Nemotron-VLM-Dataset-v1, single-turn OCR/VQA over
#                 documents and charts, selected by partition
DATASET="${DATASET:-visionarena}"

# Step 1 reads an already-downloaded dataset and never hits the network. Leave
# this empty to use the HuggingFace cache ($HF_HOME, else ~/.cache/huggingface),
# or point it at the directory holding the dataset. Populate the cache with
#   hf download lmarena-ai/VisionArena-Chat --repo-type dataset
#   hf download nvidia/Llama-Nemotron-VLM-Dataset-v1 --repo-type dataset
DATASET_PATH="${DATASET_PATH:-}"

# --- visionarena only ---
EXPORT_LIMIT="${EXPORT_LIMIT:-5000}"        # conversations to export
MAX_TURNS="${MAX_TURNS:-2}"                 # cap user turns per conversation
# ${VAR-default} rather than ${VAR:-default}: only about half the dataset is
# English, so EXPORT_LANGUAGE= has to be able to mean "keep every language".
# With the colon form an empty value would fall back to English instead, and the
# other half would stay unreachable.
EXPORT_LANGUAGE="${EXPORT_LANGUAGE-English}"       # e.g. English; empty keeps all languages
# Set EXPORT_ALLOW_DOWNLOAD non-empty to stream from the Hub instead of reading
# a local copy. Only visionarena can stream; nemotron images come from TAR
# shards that have to be on disk.
EXPORT_ALLOW_DOWNLOAD="${EXPORT_ALLOW_DOWNLOAD:-}"

# --- nemotron only ---
# Comma-separated partitions, e.g. ocr_1,ocr_4,vqa_9. Empty uses every partition
# whose images are present in the local copy. Note that the repo ships all 21
# partitions' JSONL but only some partitions' images -- run
#   python3 scripts/export_nemotron_vlm.py --list-partitions
# to see which are usable before picking.
NEMOTRON_PARTITIONS="${NEMOTRON_PARTITIONS:-}"
# Fraction of each partition to keep: 1 = 100%, 0.5 = 50%, 0.01 = 1%. Selection
# is nested, so raising this later keeps every row already exported and adds to
# it rather than resampling a different subset and re-extracting its images.
EXPORT_FRACTION="${EXPORT_FRACTION:-0.01}"
# Partitions whose images are not shipped but can be fetched from OpenImages
# (captioning_1, captioning_2, vqa_1, vqa_2, vqa_3) are downloaded in step 0.
# The download uses the same EXPORT_FRACTION and seed as the export, so only the
# images the export will actually ask for are fetched -- at fraction 1 vqa_1
# alone is 378 GB, so this matters. Set NEMOTRON_SKIP_DOWNLOAD non-empty to
# require the images be present already and fail instead of fetching them.
NEMOTRON_SKIP_DOWNLOAD="${NEMOTRON_SKIP_DOWNLOAD:-}"
# Simultaneous image requests during that download.
NEMOTRON_DOWNLOAD_CONCURRENCY="${NEMOTRON_DOWNLOAD_CONCURRENCY:-64}"
# Sampling seed. Shared by the download and the export so the two agree on
# which rows are in play; changing it selects a different subset of the same
# size and orphans the images already fetched.
EXPORT_SEED="${EXPORT_SEED:-0}"

# Cap on training rows kept after preprocessing. Leave empty to keep everything
# the export produced, which is what you want when EXPORT_FRACTION is already
# the size knob -- hence ${VAR-default}, so an empty value survives instead of
# falling back to the default.
MAX_SAMPLES="${MAX_SAMPLES-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"

# Checkpointing. 1 saves at the end of every epoch; values below 1 save within
# an epoch (0.25 = four times per epoch), which is what you want when a single
# epoch is long enough that losing one is expensive. Resuming is automatic.
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
# Set to any non-empty value to keep only the best-validation-loss checkpoint.
# Off by default, so every epoch is kept and nothing is pruned -- watch the disk.
SAVE_BEST="${SAVE_BEST:-}"

# Regeneration
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
# Set non-empty to skip steps 2-4 outright and train on whatever conversations
# already exist. Use it to continue an interrupted training run: regenerating a
# few failed stragglers would change the conversation count, which invalidates
# the prepared dataset and costs a full re-render of every row.
SKIP_REGEN="${SKIP_REGEN:-}"

# DFlash-specific parameters (best-practices recipe from RFC #979)
SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_ANCHORS="${MAX_ANCHORS:-3072}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"  # needs --loss-fn ce
LOSS_FN="${LOSS_FN:-ce}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
# Qwen2.5-VL-7B has 28 text layers, so the defaults are [2, 14, 25]; launch_vllm.py
# appends layer 28. Must match vLLM's eagle_aux_hidden_state_layer_ids.
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"

# Bound how many tokens each image expands to. Qwen2.5-VL's own default
# (max_pixels=12845056) lets a single large image consume most of the context.
# Both servers use the same value so responses are generated at the resolution
# the drafter is later trained on.
# Braces are escaped so the literal JSON does not close the ${...:-} expansion.
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\": 1003520\}}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4\}}"

# GPU assignments. Defaults assume 4 visible GPUs.
REGEN_GPUS="${REGEN_GPUS:-0,1,2,3}"         # no training runs concurrently
# Regeneration splits REGEN_GPUS the same way the extraction server does, so
# REGEN_DP * REGEN_TP must equal the REGEN_GPUS count. Tensor parallelism is
# bounded by the model: REGEN_TP has to divide the target's attention-head count
# (28 for Qwen2.5-VL-7B, so 1, 2 or 4 -- not 8). Past that bound, add replicas
# with REGEN_DP instead, or the extra GPUs sit idle through the longest step.
REGEN_DP="${REGEN_DP:-1}"
REGEN_TP="${REGEN_TP:-4}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"         # online training needs separate GPUs
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"             # for the server and the trainer
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
# The extraction server splits EXTRACT_GPUS between data and tensor parallelism,
# so EXTRACT_DP_SIZE * EXTRACT_TP must equal the EXTRACT_GPUS count. If vLLM
# refuses data parallelism for this model, swap to EXTRACT_DP_SIZE=1 EXTRACT_TP=2.
EXTRACT_DP_SIZE="${EXTRACT_DP_SIZE:-2}"
EXTRACT_TP="${EXTRACT_TP:-1}"
# =======================================

IMAGE_DIR="$OUTPUT_DIR/images"
PROMPTS_FILE="$OUTPUT_DIR/prompts.jsonl"
CONVERSATIONS_FILE="$OUTPUT_DIR/conversations.jsonl"
DATA_DIR="$OUTPUT_DIR/prepared"
# Kept outside DATA_DIR: prepare_data.py --overwrite refuses to run against a
# directory holding anything it did not write itself.
PREPARE_STAMP="$OUTPUT_DIR/prepared.stamp"
# Separate knob so a rerun at a larger scale can train into a clean directory
# while still reusing the exported prompts, images and regenerated responses.
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"

SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo "Stopping vLLM server (pid $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT

# Only the two exports exist, and an unrecognized name would otherwise leave
# EXPORT_SCRIPT unset and fail inside step 1 as an unbound-variable error.
check_dataset_config() {
    case "$DATASET" in
        visionarena) ;;
        nemotron)
            if [[ -n "$EXPORT_ALLOW_DOWNLOAD" ]]; then
                echo "EXPORT_ALLOW_DOWNLOAD does not apply to DATASET=nemotron:" >&2
                echo "its images come from TAR shards that must already be on" >&2
                echo "disk. Download the dataset first:" >&2
                echo "    hf download nvidia/Llama-Nemotron-VLM-Dataset-v1 --repo-type dataset" >&2
                exit 1
            fi
            ;;
        *)
            echo "DATASET=$DATASET is not recognized; use 'visionarena' or 'nemotron'." >&2
            exit 1
            ;;
    esac
}

# What the prepared dataset's contents depend on. prepare_data.py decides whether
# to skip purely on the presence of *.arrow, so raising MAX_SAMPLES against an
# existing directory is otherwise ignored in silence and training keeps using the
# smaller dataset -- along with a token_freq.pt built from it.
#
# The export settings are included because the conversation count alone cannot
# separate two different selections of the same size: switching partitions or
# changing the fraction at a fixed total would reuse a dataset built from other
# rows.
prepare_signature() {
    local num_conversations=0
    if [[ -f "$CONVERSATIONS_FILE" ]]; then
        num_conversations=$(wc -l < "$CONVERSATIONS_FILE" | tr -d '[:space:]')
    fi
    local source="dataset=$DATASET"
    case "$DATASET" in
        visionarena)
            source+=" limit=$EXPORT_LIMIT turns=$MAX_TURNS lang=${EXPORT_LANGUAGE:-all}"
            ;;
        nemotron)
            source+=" partitions=${NEMOTRON_PARTITIONS:-all} fraction=$EXPORT_FRACTION"
            source+=" seed=$EXPORT_SEED"
            ;;
    esac
    echo "$source max_samples=${MAX_SAMPLES:-all} seq_length=$SEQ_LENGTH conversations=$num_conversations"
}

# The trainer resumes by starting at last_checkpoint_epoch + 1, so once that
# reaches EPOCHS the epoch loop is an empty range: training does no work, saves
# nothing, and exits successfully. Refuse to launch into that.
check_training_would_run() {
    local last_epoch=-1 name
    [[ -d "$CHECKPOINT_DIR" ]] || return 0

    for path in "$CHECKPOINT_DIR"/*; do
        name=$(basename "$path")
        # Skip checkpoint_best and any non-epoch directory such as 'interrupted'.
        if [[ -d "$path" && ! -L "$path" && "$name" =~ ^[0-9]+$ ]]; then
            if (( 10#$name > last_epoch )); then
                last_epoch=$((10#$name))
            fi
        fi
    done

    (( last_epoch >= 0 )) || return 0
    if (( last_epoch + 1 >= EPOCHS )); then
        echo "Checkpoints in $CHECKPOINT_DIR already cover epoch $last_epoch, so" >&2
        echo "training would resume at epoch $((last_epoch + 1)) with EPOCHS=$EPOCHS" >&2
        echo "and silently do nothing. Raise EPOCHS, set CHECKPOINT_DIR to a fresh" >&2
        echo "path, or remove the existing checkpoints." >&2
        exit 1
    fi
}

wait_for_vllm() {
    echo "Waiting for vLLM server to be ready..."
    until curl -sf "http://localhost:${SERVER_PORT}/health" > /dev/null 2>&1; do
        # If the server died during startup, stop waiting forever.
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM server exited during startup." >&2
            exit 1
        fi
        sleep 5
    done
    echo "vLLM server ready."
}

# Fail now rather than after the export and a model load: the GPU assignments
# above index into CUDA_VISIBLE_DEVICES, so too small an allocation surfaces as
# a confusing vLLM error many minutes in.
check_gpus() {
    local available
    available=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [[ "$available" -eq 0 ]]; then
        echo "No GPUs visible; this pipeline needs them for every step after step 1." >&2
        exit 1
    fi

    local num_extract num_train num_regen
    num_extract=$(tr ',' ' ' <<< "$EXTRACT_GPUS" | wc -w)
    num_train=$(tr ',' ' ' <<< "$TRAIN_GPUS" | wc -w)
    num_regen=$(tr ',' ' ' <<< "$REGEN_GPUS" | wc -w)

    local needed=$(( num_extract + num_train ))
    if [[ "$available" -lt "$needed" ]]; then
        echo "Need $needed GPUs for EXTRACT_GPUS + TRAIN_GPUS but only $available are visible." >&2
        echo "Set EXTRACT_GPUS, TRAIN_GPUS, NUM_TRAIN_GPUS, EXTRACT_DP_SIZE, REGEN_GPUS and" >&2
        echo "REGEN_TP to match your allocation." >&2
        exit 1
    fi

    # Step 7 runs the trainer against the live server, so the two cannot share a
    # device: they would contend for memory and for the same SMs.
    local overlap=""
    for gpu in $(tr ',' ' ' <<< "$EXTRACT_GPUS"); do
        for other in $(tr ',' ' ' <<< "$TRAIN_GPUS"); do
            [[ "$gpu" == "$other" ]] && overlap+=" $gpu"
        done
    done
    if [[ -n "$overlap" ]]; then
        echo "EXTRACT_GPUS ($EXTRACT_GPUS) and TRAIN_GPUS ($TRAIN_GPUS) both use:$overlap" >&2
        echo "Online training runs the extraction server and the trainer at the same" >&2
        echo "time, so they need disjoint devices." >&2
        exit 1
    fi

    # A parallelism product below the GPU count is not an error to vLLM or
    # torchrun -- they just use fewer devices -- so it would otherwise show up
    # only as an unexplained shortfall in throughput.
    if (( REGEN_DP * REGEN_TP != num_regen )); then
        echo "REGEN_DP($REGEN_DP) * REGEN_TP($REGEN_TP) = $(( REGEN_DP * REGEN_TP )), but" >&2
        echo "REGEN_GPUS lists $num_regen GPUs. Regeneration would leave the rest idle." >&2
        exit 1
    fi
    if (( EXTRACT_DP_SIZE * EXTRACT_TP != num_extract )); then
        echo "EXTRACT_DP_SIZE($EXTRACT_DP_SIZE) * EXTRACT_TP($EXTRACT_TP) = $(( EXTRACT_DP_SIZE * EXTRACT_TP ))," >&2
        echo "but EXTRACT_GPUS lists $num_extract GPUs. The extras would sit idle." >&2
        exit 1
    fi
    if (( NUM_TRAIN_GPUS != num_train )); then
        echo "NUM_TRAIN_GPUS($NUM_TRAIN_GPUS) does not match the $num_train GPUs in" >&2
        echo "TRAIN_GPUS ($TRAIN_GPUS); torchrun would size the job to the wrong count." >&2
        exit 1
    fi

    echo "$available GPUs visible: extract=$EXTRACT_GPUS (dp$EXTRACT_DP_SIZE x tp$EXTRACT_TP), train=$TRAIN_GPUS, regen=$REGEN_GPUS (dp$REGEN_DP x tp$REGEN_TP)"
}

check_dataset_config

echo "=== Configuration ==="
echo "  model=$MODEL dataset=$DATASET max_samples=${MAX_SAMPLES:-all}"
echo "  epochs=$EPOCHS lr=$LR seq_length=$SEQ_LENGTH"
echo "  output_dir=$OUTPUT_DIR port=$SERVER_PORT"
case "$DATASET" in
    visionarena)
        echo "  export_limit=$EXPORT_LIMIT max_turns=$MAX_TURNS language=${EXPORT_LANGUAGE:-all}"
        ;;
    nemotron)
        echo "  partitions=${NEMOTRON_PARTITIONS:-all with images} fraction=$EXPORT_FRACTION seed=$EXPORT_SEED"
        if [[ -n "$NEMOTRON_SKIP_DOWNLOAD" ]]; then
            echo "  image_download=disabled (images must already be present)"
        fi
        ;;
esac
if [[ -n "$EXPORT_ALLOW_DOWNLOAD" ]]; then
    echo "  source=streaming from the Hub (downloads shards)"
else
    echo "  source=${DATASET_PATH:-HuggingFace cache (${HF_HOME:-~/.cache/huggingface})}"
fi
echo "  checkpoint_dir=$CHECKPOINT_DIR checkpoint_freq=$CHECKPOINT_FREQ"
check_gpus
# Checked here rather than at step 7 so a run that could not train anything fails
# now, instead of after the export, regeneration and preprocessing.
check_training_would_run

mkdir -p "$OUTPUT_DIR"

# Step 0: Fetch images for nemotron partitions that ship without them.
# Only the OpenImages-backed partitions can be fetched; the others are rejected
# by the downloader with a pointer to where their images live. Images land in
# $IMAGE_DIR/<partition>_images/, keeping them under the one directory vLLM is
# later given as --allowed-local-media-path.
if [[ "$DATASET" == "nemotron" && -z "$NEMOTRON_SKIP_DOWNLOAD" && -n "$NEMOTRON_PARTITIONS" ]]; then
    DOWNLOADABLE=""
    for partition in ${NEMOTRON_PARTITIONS//,/ }; do
        case "$partition" in
            captioning_1|captioning_2|vqa_1|vqa_2|vqa_3)
                DOWNLOADABLE+="${DOWNLOADABLE:+,}$partition"
                ;;
        esac
    done
    if [[ -n "$DOWNLOADABLE" ]]; then
        echo "=== Step 0: Downloading images for $DOWNLOADABLE ==="
        DOWNLOAD_ARGS=(
            --partitions "$DOWNLOADABLE"
            --fraction "$EXPORT_FRACTION"
            --seed "$EXPORT_SEED"
            --image-source "$IMAGE_DIR"
            --concurrency "$NEMOTRON_DOWNLOAD_CONCURRENCY"
        )
        if [[ -n "$DATASET_PATH" ]]; then
            DOWNLOAD_ARGS+=(--dataset-path "$DATASET_PATH")
        fi
        mkdir -p "$IMAGE_DIR"
        python3 scripts/download_nemotron_images.py "${DOWNLOAD_ARGS[@]}"
    fi
fi

# Step 1: Export prompts and materialize their images (CPU only)
# Reads the local copy of the dataset -- the HuggingFace cache by default, or
# DATASET_PATH -- so this step needs no network. Both exports write the same
# prompt-only conversations, which is why steps 2-7 need no knowledge of which
# dataset produced them.
echo "=== Step 1: Exporting $DATASET prompts and images ==="
EXPORT_ARGS=(
    --image-dir "$IMAGE_DIR"
    --outfile "$PROMPTS_FILE"
    --resume
)
case "$DATASET" in
    visionarena)
        EXPORT_SCRIPT=scripts/export_visionarena.py
        EXPORT_ARGS+=(--limit "$EXPORT_LIMIT" --max-turns "$MAX_TURNS")
        if [[ -n "$EXPORT_LANGUAGE" ]]; then
            EXPORT_ARGS+=(--language "$EXPORT_LANGUAGE")
        fi
        if [[ -n "$EXPORT_ALLOW_DOWNLOAD" ]]; then
            EXPORT_ARGS+=(--allow-download)
        elif [[ -n "$DATASET_PATH" ]]; then
            EXPORT_ARGS+=(--dataset-path "$DATASET_PATH")
        fi
        ;;
    nemotron)
        EXPORT_SCRIPT=scripts/export_nemotron_vlm.py
        # --image-source adds $IMAGE_DIR to the places images are looked for,
        # so step 0's downloads are found without hiding the TAR shards that
        # ship beside the dataset's JSONL.
        EXPORT_ARGS+=(
            --fraction "$EXPORT_FRACTION"
            --seed "$EXPORT_SEED"
            --image-source "$IMAGE_DIR"
        )
        if [[ -n "$NEMOTRON_PARTITIONS" ]]; then
            EXPORT_ARGS+=(--partitions "$NEMOTRON_PARTITIONS")
        fi
        if [[ -n "$DATASET_PATH" ]]; then
            EXPORT_ARGS+=(--dataset-path "$DATASET_PATH")
        fi
        ;;
esac
python3 "$EXPORT_SCRIPT" "${EXPORT_ARGS[@]}"

# Steps 2-4 only matter if something still needs regenerating. Ask before
# committing to a model load: on a resubmitted run the answer is usually zero,
# and step 3 cannot work this out for itself because it reads the served model
# id off the endpoint before it looks at how much work is left.
if [[ -n "$SKIP_REGEN" ]]; then
    REMAINING=0
    echo "SKIP_REGEN set; leaving $CONVERSATIONS_FILE untouched."
else
    REMAINING=$(python3 scripts/regenerate_vlm_responses.py \
        --data "$PROMPTS_FILE" \
        --outfile "$CONVERSATIONS_FILE" \
        --resume \
        --count-remaining)
    echo "Conversations still needing regeneration: $REMAINING"
    # Topping up a handful of stragglers is rarely worth what it triggers: the
    # conversation count changes, which invalidates the prepared dataset and
    # forces a full re-render of every row to recover a fraction of a percent.
    # Failures that survived their retries are usually permanent anyway.
    if (( REMAINING > 0 )) && [[ -f "$PREPARE_STAMP" ]]; then
        echo "Note: regenerating these will invalidate $DATA_DIR and rebuild it" >&2
        echo "from scratch. Set SKIP_REGEN=1 to keep the existing data and go" >&2
        echo "straight to training." >&2
    fi
fi

if [[ "$REMAINING" -gt 0 ]]; then
    # Step 2: Launch a plain vLLM server for response regeneration
    echo "=== Step 2: Launching vLLM server for regeneration ==="
    CUDA_VISIBLE_DEVICES="$REGEN_GPUS" vllm serve "$MODEL" \
        --port "$SERVER_PORT" \
        --data-parallel-size "$REGEN_DP" \
        --tensor-parallel-size "$REGEN_TP" \
        --max-model-len "$SEQ_LENGTH" \
        --allowed-local-media-path "$(realpath "$IMAGE_DIR")" \
        --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
        --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" &
    SERVER_PID=$!
    wait_for_vllm

    # Step 3: Regenerate responses with the target model (this is what makes the
    # data on-policy; the arena's original responses are discarded)
    echo "=== Step 3: Regenerating on-policy responses ==="
    python3 scripts/regenerate_vlm_responses.py \
        --data "$PROMPTS_FILE" \
        --outfile "$CONVERSATIONS_FILE" \
        --endpoint "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        --concurrency "$REGEN_CONCURRENCY" \
        --max-tokens "$REGEN_MAX_TOKENS" \
        --resume

    # Step 4: Stop the regeneration server before starting the hidden-states one
    echo "=== Step 4: Stopping regeneration server ==="
    cleanup
else
    echo "=== Steps 2-4: Skipped, regeneration already complete ==="
fi

# Step 5: Launch vLLM configured for hidden-state extraction
echo "=== Step 5: Launching vLLM server for hidden-state extraction ==="
CUDA_VISIBLE_DEVICES="$EXTRACT_GPUS" python3 scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$EXTRACT_DP_SIZE" \
       --tensor-parallel-size "$EXTRACT_TP" \
       --port "$SERVER_PORT" \
       --max-model-len "$SEQ_LENGTH" \
       --allowed-local-media-path "$(realpath "$IMAGE_DIR")" \
       --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
       --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" &
SERVER_PID=$!
wait_for_vllm

# save_to_disk writes the Arrow shards before the state.json that ties them
# together, so a run killed inside it leaves shards that are not a loadable
# dataset. prepare_data.py decides whether to skip by globbing *.arrow alone, so
# it would treat that wreckage as finished work and step 7 would train on a
# truncated dataset without complaining. Clear it and let step 6 redo the work.
if compgen -G "$DATA_DIR/*.arrow" > /dev/null && [[ ! -f "$DATA_DIR/state.json" ]]; then
    echo "Prepared dataset at $DATA_DIR has shards but no state.json; removing it."
    rm -rf "$DATA_DIR"
fi

# Same idea for a dataset that is intact but built from different inputs.
PREPARE_WANT=$(prepare_signature)
if [[ -d "$DATA_DIR" ]] && compgen -G "$DATA_DIR/*.arrow" > /dev/null; then
    if [[ ! -f "$PREPARE_STAMP" ]]; then
        # A dataset predating the stamp, so its inputs are unknown. Rebuilding on
        # a guess would throw away hours of rendering; reusing on a guess would
        # train on the wrong data. Neither is ours to choose silently.
        echo "Prepared dataset at $DATA_DIR has no stamp, so what it was built" >&2
        echo "from is unknown. Either rebuild it:" >&2
        echo "    rm -rf $DATA_DIR" >&2
        echo "or, if it already matches the settings below, adopt it:" >&2
        echo "    printf '%s\\n' '$PREPARE_WANT' > $PREPARE_STAMP" >&2
        exit 1
    fi
    PREPARE_HAVE=$(cat "$PREPARE_STAMP")
    if [[ "$PREPARE_HAVE" != "$PREPARE_WANT" ]]; then
        echo "Prepared dataset is stale, rebuilding it."
        echo "  have: $PREPARE_HAVE"
        echo "  want: $PREPARE_WANT"
        rm -rf "$DATA_DIR"
    fi
fi

# Step 6: Preprocess. The render endpoint applies the serving chat template,
# expands each image into its placeholder tokens, and derives the loss mask from
# the assistant-turn boundary.
echo "=== Step 6: Preparing data ==="
PREPARE_ARGS=(
    --model "$MODEL"
    --data "$CONVERSATIONS_FILE"
    --output "$DATA_DIR"
    --render-endpoint "http://localhost:${SERVER_PORT}"
    --seq-length "$SEQ_LENGTH"
)
# Omitted rather than passed as a sentinel: prepare_data.py treats an absent
# --max-samples as "keep everything", which is what you want when the export
# fraction is already the size knob.
if [[ -n "$MAX_SAMPLES" ]]; then
    PREPARE_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
python3 scripts/prepare_data.py "${PREPARE_ARGS[@]}"

# Recorded only now, so a preprocessing run that dies leaves no stamp and the
# next run rebuilds rather than trusting a partial dataset.
printf '%s\n' "$PREPARE_WANT" > "$PREPARE_STAMP"

# Step 7: Train against the live vLLM server.
# Checkpoints land in $CHECKPOINT_DIR/<epoch>/ with a checkpoint_best symlink.
# Rerunning resumes from the newest one automatically, and a SIGTERM (e.g. a
# scheduler walltime) saves an `interrupted` checkpoint first -- give the
# scheduler enough grace time for that write to finish.
echo "=== Step 7: Training ==="
TRAIN_ARGS=(
    --verifier-name-or-path "$MODEL"
    --data-path "$DATA_DIR"
    --vllm-endpoint "http://localhost:${SERVER_PORT}/v1"
    --save-path "$CHECKPOINT_DIR"
    --draft-vocab-size "$DRAFT_VOCAB_SIZE"
    --epochs "$EPOCHS"
    --lr "$LR"
    --total-seq-len "$SEQ_LENGTH"
    --speculator-type "$SPECULATOR_TYPE"
    --block-size "$BLOCK_SIZE"
    --max-anchors "$MAX_ANCHORS"
    --num-layers "$NUM_LAYERS"
    --per-position-loss-weight "$PER_POSITION_LOSS_WEIGHT"
    --loss-fn "$LOSS_FN"
    --checkpoint-freq "$CHECKPOINT_FREQ"
    --on-missing generate
    --on-generate delete
)
if [[ -n "$SAVE_BEST" ]]; then
    TRAIN_ARGS+=(--save-best)
fi
# TARGET_LAYER_IDS is intentionally unquoted: it is a space-separated list.
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py \
    "${TRAIN_ARGS[@]}" \
    --target-layer-ids $TARGET_LAYER_IDS

echo "Done. Checkpoints saved to $CHECKPOINT_DIR/"
