#!/bin/bash
# Standalone online DFlash training for Qwen2.5-VL-7B on
# HuggingFaceM4/the_cauldron. It exports prompt-only conversations, regenerates
# target-model responses, renders multimodal training data, and trains online.
#
# CAULDRON_SUBSETS is a comma list; empty selects all official subsets.
# PERC_SAMPLES is one deterministic fraction applied to every subset. It may
# grow in an existing OUTPUT_DIR, but cannot shrink; use MAX_SAMPLES to cap a
# training run without discarding the reusable conversation pool.

set -euo pipefail
unset VLLM_PORT VLLM_DP_SIZE

# Data and model
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen2_5_vl_7b_cauldron}"
CAULDRON_DATASET_PATH="${CAULDRON_DATASET_PATH:-}"
CAULDRON_SUBSETS="${CAULDRON_SUBSETS:-}"
CAULDRON_ALLOW_DOWNLOAD="${CAULDRON_ALLOW_DOWNLOAD:-}"
PERC_SAMPLES="${PERC_SAMPLES:-0.1}"
EXPORT_SEED="${EXPORT_SEED:-0}"
MAX_SAMPLES="${MAX_SAMPLES-5000}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"

# Regeneration
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
SKIP_REGEN="${SKIP_REGEN:-}"

# Training
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
SAVE_BEST="${SAVE_BEST:-}"
SPECULATOR_TYPE="${SPECULATOR_TYPE:-dflash}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_ANCHORS="${MAX_ANCHORS:-3072}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"
LOSS_FN="${LOSS_FN:-ce}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"

# Serving and GPU layout
SERVER_PORT="${SERVER_PORT:-8000}"
MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\": 1003520\}}"
LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\": 4\}}"
REGEN_GPUS="${REGEN_GPUS:-0,1,2,3}"
REGEN_DP="${REGEN_DP:-1}"
REGEN_TP="${REGEN_TP:-4}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
EXTRACT_DP_SIZE="${EXTRACT_DP_SIZE:-2}"
EXTRACT_TP="${EXTRACT_TP:-1}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"

IMAGE_DIR="$OUTPUT_DIR/images"
PROMPTS_FILE="$OUTPUT_DIR/prompts.jsonl"
CONVERSATIONS_FILE="$OUTPUT_DIR/conversations.jsonl"
POOL_MANIFEST="$OUTPUT_DIR/cauldron_pool.json"
DATA_DIR="$OUTPUT_DIR/prepared"
PREPARE_STAMP="$OUTPUT_DIR/prepared.stamp"
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

line_count() {
    local path="$1"
    if [[ -f "$path" ]]; then
        wc -l < "$path" | tr -d '[:space:]'
    else
        echo 0
    fi
}

check_config() {
    if ! [[ "$PERC_SAMPLES" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] \
        || ! awk -v value="$PERC_SAMPLES" \
            'BEGIN { exit !(value > 0 && value <= 1) }'; then
        echo "PERC_SAMPLES=$PERC_SAMPLES is invalid; use one value in (0, 1]." >&2
        exit 1
    fi
    if [[ -n "$MAX_SAMPLES" ]] \
        && { ! [[ "$MAX_SAMPLES" =~ ^[0-9]+$ ]] || (( 10#$MAX_SAMPLES <= 0 )); }; then
        echo "MAX_SAMPLES=$MAX_SAMPLES is invalid; use a positive integer or empty." >&2
        exit 1
    fi
    if [[ -n "$CAULDRON_DATASET_PATH" && -n "$CAULDRON_ALLOW_DOWNLOAD" ]]; then
        echo "CAULDRON_DATASET_PATH and CAULDRON_ALLOW_DOWNLOAD are mutually exclusive." >&2
        exit 1
    fi
}

check_output_dir() {
    if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null || [[ ! -w "$OUTPUT_DIR" ]]; then
        echo "OUTPUT_DIR=$OUTPUT_DIR cannot be created or is not writable." >&2
        exit 1
    fi
    local avail_kb
    avail_kb=$(df -Pk "$OUTPUT_DIR" 2>/dev/null | awk 'NR==2 {print $4}')
    if [[ -n "$avail_kb" ]]; then
        echo "  output_dir has $(( avail_kb / 1024 / 1024 )) GB free"
    fi
}

check_gpus() {
    local available num_regen num_extract num_train overlap="" gpu_list
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is unavailable; cannot validate the GPU allocation." >&2
        exit 1
    fi
    if ! gpu_list=$(nvidia-smi --list-gpus 2>/dev/null); then
        echo "nvidia-smi failed; cannot validate the GPU allocation." >&2
        exit 1
    fi
    available=$(printf '%s\n' "$gpu_list" | awk 'NF {count++} END {print count+0}')
    if (( available == 0 )); then
        echo "No GPUs visible; regeneration and training require GPUs." >&2
        exit 1
    fi
    local label value gpu seen
    for label in REGEN_GPUS EXTRACT_GPUS TRAIN_GPUS; do
        value="${!label}"
        if ! [[ "$value" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
            echo "$label=$value is invalid; use comma-separated visible GPU indices." >&2
            exit 1
        fi
        seen=","
        for gpu in ${value//,/ }; do
            if (( 10#$gpu >= available )); then
                echo "$label references GPU $gpu, but only indices 0-$((available - 1)) are visible." >&2
                exit 1
            fi
            if [[ "$seen" == *",$gpu,"* ]]; then
                echo "$label lists GPU $gpu more than once." >&2
                exit 1
            fi
            seen+="$gpu,"
        done
    done
    num_regen=$(tr ',' ' ' <<< "$REGEN_GPUS" | wc -w)
    num_extract=$(tr ',' ' ' <<< "$EXTRACT_GPUS" | wc -w)
    num_train=$(tr ',' ' ' <<< "$TRAIN_GPUS" | wc -w)
    if (( available < num_extract + num_train )); then
        echo "Need $(( num_extract + num_train )) visible GPUs for online training; found $available." >&2
        exit 1
    fi
    for gpu in $(tr ',' ' ' <<< "$EXTRACT_GPUS"); do
        for other in $(tr ',' ' ' <<< "$TRAIN_GPUS"); do
            [[ "$gpu" == "$other" ]] && overlap+=" $gpu"
        done
    done
    if [[ -n "$overlap" ]]; then
        echo "EXTRACT_GPUS and TRAIN_GPUS overlap on:$overlap" >&2
        exit 1
    fi
    if (( REGEN_DP * REGEN_TP != num_regen )); then
        echo "REGEN_DP * REGEN_TP must equal the REGEN_GPUS count ($num_regen)." >&2
        exit 1
    fi
    if (( EXTRACT_DP_SIZE * EXTRACT_TP != num_extract )); then
        echo "EXTRACT_DP_SIZE * EXTRACT_TP must equal the EXTRACT_GPUS count ($num_extract)." >&2
        exit 1
    fi
    if (( NUM_TRAIN_GPUS != num_train )); then
        echo "NUM_TRAIN_GPUS must equal the TRAIN_GPUS count ($num_train)." >&2
        exit 1
    fi
    echo "$available GPUs visible: regen=$REGEN_GPUS extract=$EXTRACT_GPUS train=$TRAIN_GPUS"
}

check_training_would_run() {
    local last_epoch=-1 path name
    [[ -d "$CHECKPOINT_DIR" ]] || return 0
    for path in "$CHECKPOINT_DIR"/*; do
        name=$(basename "$path")
        if [[ -d "$path" && ! -L "$path" && "$name" =~ ^[0-9]+$ ]] \
            && (( 10#$name > last_epoch )); then
            last_epoch=$((10#$name))
        fi
    done
    if (( last_epoch >= 0 && last_epoch + 1 >= EPOCHS )); then
        echo "Checkpoints already cover epoch $last_epoch; EPOCHS=$EPOCHS would do no work." >&2
        echo "Raise EPOCHS or use a fresh CHECKPOINT_DIR." >&2
        exit 1
    fi
}

wait_for_vllm() {
    echo "Waiting for vLLM server..."
    until curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "vLLM exited during startup." >&2
            exit 1
        fi
        sleep 5
    done
}

check_prompt_cap() {
    [[ -n "$MAX_SAMPLES" ]] || return 0
    local available
    available=$(line_count "$PROMPTS_FILE")
    if (( 10#$MAX_SAMPLES > available )); then
        python3 - "$PERC_SAMPLES" "$available" "$MAX_SAMPLES" <<'PY' >&2
import sys
from scripts.export_cauldron import format_sample_cap_error

print(format_sample_cap_error(float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])))
PY
        exit 1
    fi
    echo "  prompt_pool=$available conversations; training cap=$MAX_SAMPLES"
}

check_successful_cap() {
    [[ -n "$MAX_SAMPLES" ]] || return 0
    local successful
    successful=$(line_count "$CONVERSATIONS_FILE")
    if (( successful < 10#$MAX_SAMPLES )); then
        echo "Only $successful conversations regenerated successfully, below MAX_SAMPLES=$MAX_SAMPLES." >&2
        echo "Pending rows (including any skipped with SKIP_REGEN) and generation" >&2
        echo "failures are excluded. Inspect ${CONVERSATIONS_FILE%.jsonl}.errors.jsonl," >&2
        echo "then retry regeneration or lower MAX_SAMPLES." >&2
        exit 1
    fi
}

prepare_signature() {
    local pool_count conversation_count manifest_hash="missing"
    pool_count=$(line_count "$PROMPTS_FILE")
    conversation_count=$(line_count "$CONVERSATIONS_FILE")
    if [[ -f "$POOL_MANIFEST" ]]; then
        manifest_hash=$(sha256sum "$POOL_MANIFEST" | awk '{print $1}')
    fi
    echo "pool_manifest=$manifest_hash pool_count=$pool_count conversations=$conversation_count max_samples=${MAX_SAMPLES:-all} seq_length=$SEQ_LENGTH"
}

check_config
check_output_dir
check_gpus
check_training_would_run

echo "=== Configuration ==="
echo "  model=$MODEL subsets=${CAULDRON_SUBSETS:-all 50}"
echo "  perc_samples=$PERC_SAMPLES max_samples=${MAX_SAMPLES:-all} seed=$EXPORT_SEED"
echo "  output_dir=$OUTPUT_DIR checkpoint_dir=$CHECKPOINT_DIR"

# Step 1: deterministic append-only prompt pool export.
echo "=== Step 1: Exporting Cauldron prompts and images ==="
EXPORT_ARGS=(
    --image-dir "$IMAGE_DIR"
    --outfile "$PROMPTS_FILE"
    --pool-manifest "$POOL_MANIFEST"
    --fraction "$PERC_SAMPLES"
    --seed "$EXPORT_SEED"
    --resume
)
if [[ -n "$CAULDRON_SUBSETS" ]]; then
    EXPORT_ARGS+=(--subsets "$CAULDRON_SUBSETS")
fi
if [[ -n "$CAULDRON_ALLOW_DOWNLOAD" ]]; then
    EXPORT_ARGS+=(--allow-download)
elif [[ -n "$CAULDRON_DATASET_PATH" ]]; then
    EXPORT_ARGS+=(--dataset-path "$CAULDRON_DATASET_PATH")
fi
python3 scripts/export_cauldron.py "${EXPORT_ARGS[@]}"

# This check is intentionally before either model is loaded.
check_prompt_cap
ALLOWED_MEDIA_PATH=$(realpath "$IMAGE_DIR")
echo "  allowed_local_media_path=$ALLOWED_MEDIA_PATH"

# Load the regeneration model only when exported IDs are still missing.
if [[ -n "$SKIP_REGEN" ]]; then
    REMAINING=0
    echo "SKIP_REGEN set; leaving $CONVERSATIONS_FILE untouched."
else
    REMAINING=$(python3 scripts/regenerate_vlm_responses.py \
        --data "$PROMPTS_FILE" --outfile "$CONVERSATIONS_FILE" \
        --resume --count-remaining)
    echo "Conversations still needing regeneration: $REMAINING"
    if (( REMAINING > 0 )) && [[ -f "$PREPARE_STAMP" ]]; then
        echo "Regeneration growth will invalidate and rebuild $DATA_DIR." >&2
    fi
fi

if (( REMAINING > 0 )); then
    echo "=== Step 2: Launching regeneration server ==="
    CUDA_VISIBLE_DEVICES="$REGEN_GPUS" vllm serve "$MODEL" \
        --port "$SERVER_PORT" \
        --data-parallel-size "$REGEN_DP" \
        --tensor-parallel-size "$REGEN_TP" \
        --max-model-len "$SEQ_LENGTH" \
        --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
        --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
        --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" &
    SERVER_PID=$!
    wait_for_vllm

    echo "=== Step 3: Regenerating responses ==="
    python3 scripts/regenerate_vlm_responses.py \
        --data "$PROMPTS_FILE" \
        --outfile "$CONVERSATIONS_FILE" \
        --endpoint "http://localhost:${SERVER_PORT}/v1/chat/completions" \
        --concurrency "$REGEN_CONCURRENCY" \
        --max-tokens "$REGEN_MAX_TOKENS" \
        --resume
    echo "=== Step 4: Stopping regeneration server ==="
    cleanup
else
    echo "=== Steps 2-4: Regeneration already complete ==="
fi
check_successful_cap

echo "=== Step 5: Launching hidden-state server ==="
CUDA_VISIBLE_DEVICES="$EXTRACT_GPUS" python3 scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    -- --data-parallel-size "$EXTRACT_DP_SIZE" \
       --tensor-parallel-size "$EXTRACT_TP" \
       --port "$SERVER_PORT" \
       --max-model-len "$SEQ_LENGTH" \
       --allowed-local-media-path "$ALLOWED_MEDIA_PATH" \
       --mm-processor-kwargs "$MM_PROCESSOR_KWARGS" \
       --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT" &
SERVER_PID=$!
wait_for_vllm

# Remove interrupted save_to_disk output that prepare_data.py could mistake for
# a complete dataset merely because Arrow shards exist.
if compgen -G "$DATA_DIR/*.arrow" >/dev/null && [[ ! -f "$DATA_DIR/state.json" ]]; then
    echo "Removing partial prepared dataset at $DATA_DIR."
    rm -rf "$DATA_DIR"
fi

PREPARE_WANT=$(prepare_signature)
if [[ -d "$DATA_DIR" ]] && compgen -G "$DATA_DIR/*.arrow" >/dev/null; then
    if [[ ! -f "$PREPARE_STAMP" ]]; then
        echo "$DATA_DIR exists without $PREPARE_STAMP; its inputs are unknown." >&2
        echo "Remove $DATA_DIR to rebuild, or write the expected stamp explicitly:" >&2
        echo "  printf '%s\\n' '$PREPARE_WANT' > '$PREPARE_STAMP'" >&2
        exit 1
    fi
    PREPARE_HAVE=$(<"$PREPARE_STAMP")
    if [[ "$PREPARE_HAVE" != "$PREPARE_WANT" ]]; then
        echo "Prepared dataset inputs changed; rebuilding."
        rm -rf "$DATA_DIR"
    fi
fi

echo "=== Step 6: Preparing data ==="
PREPARE_ARGS=(
    --model "$MODEL"
    --data "$CONVERSATIONS_FILE"
    --output "$DATA_DIR"
    --render-endpoint "http://localhost:${SERVER_PORT}"
    --seq-length "$SEQ_LENGTH"
)
if [[ -n "$MAX_SAMPLES" ]]; then
    PREPARE_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
python3 scripts/prepare_data.py "${PREPARE_ARGS[@]}"
printf '%s\n' "$PREPARE_WANT" > "$PREPARE_STAMP"

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
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    scripts/train.py "${TRAIN_ARGS[@]}" --target-layer-ids $TARGET_LAYER_IDS

echo "Done. Checkpoints saved to $CHECKPOINT_DIR/"
