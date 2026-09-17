#!/usr/bin/env bash
set -euo pipefail
cd /workspace/validated-checkout
export PATH=/workspace/tools/git-package/usr/bin:/workspace/venv/bin:$PATH
export MODEL=/hf-cache/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
export EXPORT_LIMIT=12 MAX_SAMPLES=12 EPOCHS=1 SEQ_LENGTH=2048 REGEN_MAX_TOKENS=128 REGEN_CONCURRENCY=4 MAX_ANCHORS=32 BLOCK_SIZE=4 NUM_LAYERS=1 NUM_WORKERS=0 REGEN_TP=1 EXTRACT_TP=1 NUM_TRAIN_GPUS=1 SERVER_EAGER=1
export MM_PROCESSOR_KWARGS='{"max_pixels":200704}'
if [[ "$1" == visionarena ]]; then
    export DATASET_PATH=/hf-cache/hub/datasets--lmarena-ai--VisionArena-Chat/snapshots/1394b4f59ab6f1f2e5aff6bc15b448e15960e170
    export OUTPUT_DIR=/workspace/visionarena-smoke-final MAX_TURNS=1 REGEN_GPUS=0 EXTRACT_GPUS=0 TRAIN_GPUS=1 SERVER_PORT=8100
else
    export REQUESTS_FILE=/workspace/requests-smoke-final.jsonl OUTPUT_DIR=/workspace/requests-smoke-final REGEN_GPUS=2 EXTRACT_GPUS=2 TRAIN_GPUS=3 SERVER_PORT=8101
    python - <<'PY'
import json
from pathlib import Path
request=json.loads(Path('examples/data/vlm_attribute_request.json').read_text())
Path('/workspace/requests-smoke-final.jsonl').write_text(''.join(json.dumps(request | {'id': f'smoke-{i}'})+'\n' for i in range(12)))
PY
fi
set +e
bash "examples/train/dflash_qwen2_5_vl_7b_${1}_online.sh" > "${OUTPUT_DIR}.log" 2>&1
result=$?
printf '%s\n' "$result" > "${OUTPUT_DIR}.exit"
exit "$result"
