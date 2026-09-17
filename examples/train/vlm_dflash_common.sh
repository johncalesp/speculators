#!/usr/bin/env bash
# Shared implementation; source from one of the two VLM launchers.
set -euo pipefail
: "${VLM_SOURCE:?Use a VisionArena or requests launcher}"
[[ -f scripts/train.py ]] || { echo 'Run from the repository root.' >&2; exit 1; }
unset VLLM_PORT VLLM_DP_SIZE
PYTHON="${PYTHON:-python3}"
VLLM_PYTHON="${VLLM_PYTHON:-$PYTHON}"
export MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export VLM_SOURCE
export DATASET_PATH="${DATASET_PATH:-}"
export REQUESTS_FILE="${REQUESTS_FILE:-}"
export EXPORT_LIMIT="${EXPORT_LIMIT:-5000}"
export MAX_TURNS="${MAX_TURNS:-2}"
export EXPORT_LANGUAGE="${EXPORT_LANGUAGE-English}"
export EXPORT_ALLOW_DOWNLOAD="${EXPORT_ALLOW_DOWNLOAD:-}"
export EXPORT_SEED="${EXPORT_SEED:-0}"
export MAX_SAMPLES="${MAX_SAMPLES-5000}"
export SEQ_LENGTH="${SEQ_LENGTH:-8192}"
export REGEN_MAX_TOKENS="${REGEN_MAX_TOKENS:-2048}"
export REGEN_SAMPLING_PARAMS="${REGEN_SAMPLING_PARAMS:-}"
[[ -n "$REGEN_SAMPLING_PARAMS" ]] || REGEN_SAMPLING_PARAMS='{}'
export MM_PROCESSOR_KWARGS="${MM_PROCESSOR_KWARGS:-{\"max_pixels\":1003520\}}"
export LIMIT_MM_PER_PROMPT="${LIMIT_MM_PER_PROMPT:-{\"image\":4\}}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-3e-4}"
CHECKPOINT_FREQ="${CHECKPOINT_FREQ:-1}"
SAVE_BEST="${SAVE_BEST:-}"
SKIP_REGEN="${SKIP_REGEN:-}"
REGEN_CONCURRENCY="${REGEN_CONCURRENCY:-32}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
MAX_ANCHORS="${MAX_ANCHORS:-3072}"
NUM_LAYERS="${NUM_LAYERS:-5}"
PER_POSITION_LOSS_WEIGHT="${PER_POSITION_LOSS_WEIGHT:-dpace}"
LOSS_FN="${LOSS_FN:-ce}"
DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-32000}"
TARGET_LAYER_IDS="${TARGET_LAYER_IDS:-2 14 25}"
NUM_WORKERS="${NUM_WORKERS:-12}"
SERVER_PORT="${SERVER_PORT:-8000}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-1800}"
SERVER_EAGER="${SERVER_EAGER:-}"
REGEN_GPUS="${REGEN_GPUS:-0,1,2,3}"
REGEN_DP="${REGEN_DP:-1}"
REGEN_TP="${REGEN_TP:-4}"
EXTRACT_GPUS="${EXTRACT_GPUS:-0,1}"
EXTRACT_DP_SIZE="${EXTRACT_DP_SIZE:-1}"
EXTRACT_TP="${EXTRACT_TP:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-2,3}"
NUM_TRAIN_GPUS="${NUM_TRAIN_GPUS:-2}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")
IMAGE_DIR="$OUTPUT_DIR/images"
PROMPTS_FILE="$OUTPUT_DIR/prompts.jsonl"
CONVERSATIONS_FILE="$OUTPUT_DIR/conversations.jsonl"
DATA_DIR="$OUTPUT_DIR/prepared"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT_DIR/checkpoints}"
HIDDEN_STATES_DIR="$OUTPUT_DIR/hidden_states"
mkdir -p "$IMAGE_DIR" "$HIDDEN_STATES_DIR"
SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Validate allocations before exporting data or loading the target.
"$PYTHON" - "$REGEN_GPUS" "$EXTRACT_GPUS" "$TRAIN_GPUS" "$REGEN_DP" "$REGEN_TP" "$EXTRACT_DP_SIZE" "$EXTRACT_TP" "$NUM_TRAIN_GPUS" <<'PY'
import subprocess
import sys
r, e, t = [value.split(',') for value in sys.argv[1:4]]
rd, rt, ed, et, nt = map(int, sys.argv[4:])
available = set(subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True).split())
for group in (r, e, t):
    if len(set(group)) != len(group) or not set(group) <= available:
        sys.exit(f'Invalid GPU list {group}; available indices: {sorted(available)}')
if set(e) & set(t):
    sys.exit('EXTRACT_GPUS and TRAIN_GPUS must be disjoint')
if min(rd, rt, ed, et, nt) < 1 or (rd * rt, ed * et, nt) != (len(r), len(e), len(t)):
    sys.exit('GPU counts must match REGEN_DP*REGEN_TP, EXTRACT_DP_SIZE*EXTRACT_TP, and NUM_TRAIN_GPUS')
PY
if curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null; then
    echo "Port $SERVER_PORT already serves a model; choose another SERVER_PORT." >&2
    exit 1
fi
# Reject completed runs early. Increasing EPOCHS resumes the existing trainer.
for path in "$CHECKPOINT_DIR"/*; do
    name=$(basename "$path")
    if [[ -d "$path" && ! -L "$path" && "$name" =~ ^[0-9]+$ ]] && (( 10#$name + 1 >= EPOCHS )); then
        echo "Training already reached EPOCHS=$EPOCHS. Raise EPOCHS or use a fresh OUTPUT_DIR." >&2
        exit 1
    fi
done

# Pin the data/model recipe so resumed responses cannot mix targets or settings.
"$PYTHON" - "$OUTPUT_DIR" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
root = Path(sys.argv[1])
keys = 'VLM_SOURCE MODEL DATASET_PATH REQUESTS_FILE EXPORT_LIMIT MAX_TURNS EXPORT_LANGUAGE EXPORT_ALLOW_DOWNLOAD EXPORT_SEED MAX_SAMPLES SEQ_LENGTH REGEN_MAX_TOKENS REGEN_SAMPLING_PARAMS MM_PROCESSOR_KWARGS LIMIT_MM_PER_PROMPT'.split()
config = {key: os.environ[key] for key in keys}
for key in ('REGEN_SAMPLING_PARAMS', 'MM_PROCESSOR_KWARGS', 'LIMIT_MM_PER_PROMPT'):
    if not isinstance(json.loads(config[key]), dict):
        sys.exit(f'{key} must be a JSON object')
if config['REQUESTS_FILE']:
    with open(config['REQUESTS_FILE'], 'rb') as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    config['requests_sha256'] = digest.hexdigest()
path = root / 'pipeline_config.json'
if path.exists():
    if json.loads(path.read_text()) != config:
        sys.exit('Data/model settings changed; use a fresh OUTPUT_DIR. See pipeline_config.json.')
elif any((root / name).exists() for name in ('prompts.jsonl', 'conversations.jsonl', 'prepared', 'mix.stamp')):
    sys.exit('Existing output has no matching pipeline_config.json; use a fresh OUTPUT_DIR.')
path.write_text(json.dumps(config, indent=2) + '\n')
PY

echo "=== 1/5: Exporting $VLM_SOURCE prompts ==="
EXPORT_ARGS=(--image-dir "$IMAGE_DIR" --outfile "$PROMPTS_FILE" --limit "$EXPORT_LIMIT")
if [[ "$VLM_SOURCE" == visionarena ]]; then
    EXPORT_ARGS+=(--resume --max-turns "$MAX_TURNS" --seed "$EXPORT_SEED")
    [[ -z "$EXPORT_LANGUAGE" ]] || EXPORT_ARGS+=(--language "$EXPORT_LANGUAGE")
    [[ -z "$DATASET_PATH" ]] || EXPORT_ARGS+=(--dataset-path "$DATASET_PATH")
    [[ -z "$EXPORT_ALLOW_DOWNLOAD" ]] || EXPORT_ARGS+=(--allow-download)
    "$PYTHON" scripts/export_visionarena.py "${EXPORT_ARGS[@]}"
else
    "$PYTHON" scripts/export_vlm_requests.py --input "$REQUESTS_FILE" "${EXPORT_ARGS[@]}"
fi
[[ -s "$PROMPTS_FILE" ]] || { echo 'No prompts exported.' >&2; exit 1; }
SERVER_ARGS=(--host 127.0.0.1 --port "$SERVER_PORT" --max-model-len "$SEQ_LENGTH"
    --allowed-local-media-path "$IMAGE_DIR" --mm-processor-kwargs "$MM_PROCESSOR_KWARGS"
    --limit-mm-per-prompt "$LIMIT_MM_PER_PROMPT")
[[ -z "$SERVER_EAGER" ]] || SERVER_ARGS+=(--enforce-eager)
wait_for_vllm() {
    local started=$SECONDS
    until curl -sf "http://localhost:$SERVER_PORT/health" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null || (( SECONDS - started > SERVER_START_TIMEOUT )); then
            echo "vLLM failed to start; see $OUTPUT_DIR/*.log" >&2
            exit 1
        fi
        sleep 5
    done
}
if [[ -z "$SKIP_REGEN" ]]; then
    REMAINING=$("$PYTHON" scripts/regenerate_vlm_responses.py --data "$PROMPTS_FILE" --outfile "$CONVERSATIONS_FILE" --resume --count-remaining)
    if (( REMAINING > 0 )); then
        echo '=== 2/5: Regenerating target responses ==='
        CUDA_VISIBLE_DEVICES="$REGEN_GPUS" "$VLLM_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL" \
            --data-parallel-size "$REGEN_DP" --tensor-parallel-size "$REGEN_TP" \
            "${SERVER_ARGS[@]}" > "$OUTPUT_DIR/regeneration_server.log" 2>&1 &
        SERVER_PID=$!
        wait_for_vllm
        "$PYTHON" scripts/regenerate_vlm_responses.py --data "$PROMPTS_FILE" --outfile "$CONVERSATIONS_FILE" \
            --endpoint "http://localhost:$SERVER_PORT/v1/chat/completions" --model "$MODEL" \
            --concurrency "$REGEN_CONCURRENCY" --max-tokens "$REGEN_MAX_TOKENS" \
            --sampling-params "$REGEN_SAMPLING_PARAMS" --resume
        cleanup
    fi
fi
# Never treat a partial regeneration as a successful customer run.
"$PYTHON" - "$PROMPTS_FILE" "$CONVERSATIONS_FILE" <<'PY'
import json
import sys
from pathlib import Path
prompts = {r['conversation_id'] for r in map(json.loads, Path(sys.argv[1]).read_text().splitlines())}
path = Path(sys.argv[2])
rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
completed = {r['conversation_id'] for r in rows}
if prompts != completed:
    sys.exit(f'Regeneration incomplete: {len(prompts - completed)} missing, {len(completed - prompts)} unexpected. Inspect conversations.errors.jsonl and rerun.')
if len(rows) < 2:
    sys.exit('Need at least two conversations for training and validation; add more input records.')
PY

echo '=== 3/5: Starting hidden-state extraction ==='
read -r -a LAYER_IDS <<< "$TARGET_LAYER_IDS"
CUDA_VISIBLE_DEVICES="$EXTRACT_GPUS" "$VLLM_PYTHON" scripts/launch_vllm.py "$MODEL" \
    --hidden-states-path "$HIDDEN_STATES_DIR" --target-layer-ids "${LAYER_IDS[@]}" \
    -- --data-parallel-size "$EXTRACT_DP_SIZE" --tensor-parallel-size "$EXTRACT_TP" \
    "${SERVER_ARGS[@]}" > "$OUTPUT_DIR/extraction_server.log" 2>&1 &
SERVER_PID=$!
wait_for_vllm

echo '=== 4/5: Preparing rendered training data ==='
# Hash content, not row count: retrying a failed generation may change the data.
PREPARE_WANT=$(sha256sum "$CONVERSATIONS_FILE" | cut -d ' ' -f1)
PREPARE_ARGS=(--model "$MODEL" --data "$CONVERSATIONS_FILE" --output "$DATA_DIR"
    --render-endpoint "http://localhost:$SERVER_PORT" --seq-length "$SEQ_LENGTH")
[[ -z "$MAX_SAMPLES" ]] || PREPARE_ARGS+=(--max-samples "$MAX_SAMPLES")
if [[ -d "$DATA_DIR" ]] && { [[ ! -f "$DATA_DIR/state.json" ]] || [[ ! -f "$OUTPUT_DIR/prepared.stamp" ]] || [[ "$(cat "$OUTPUT_DIR/prepared.stamp")" != "$PREPARE_WANT" ]]; }; then
    PREPARE_ARGS+=(--overwrite)
fi
"$PYTHON" scripts/prepare_data.py "${PREPARE_ARGS[@]}"
printf '%s\n' "$PREPARE_WANT" > "$OUTPUT_DIR/prepared.stamp"

echo '=== 5/5: Training DFlash ==='
TRAIN_ARGS=(--verifier-name-or-path "$MODEL" --data-path "$DATA_DIR"
    --vllm-endpoint "http://localhost:$SERVER_PORT/v1" --save-path "$CHECKPOINT_DIR"
    --hidden-states-path "$HIDDEN_STATES_DIR" --draft-vocab-size "$DRAFT_VOCAB_SIZE"
    --epochs "$EPOCHS" --lr "$LR" --total-seq-len "$SEQ_LENGTH" --speculator-type dflash
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" --num-layers "$NUM_LAYERS"
    --per-position-loss-weight "$PER_POSITION_LOSS_WEIGHT" --loss-fn "$LOSS_FN"
    --checkpoint-freq "$CHECKPOINT_FREQ" --num-workers "$NUM_WORKERS"
    --on-missing generate --on-generate delete --target-layer-ids "${LAYER_IDS[@]}")
[[ -z "$SAVE_BEST" ]] || TRAIN_ARGS+=(--save-best)
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" scripts/train.py "${TRAIN_ARGS[@]}"
echo "Done. Checkpoints and train_command.txt: $CHECKPOINT_DIR"
