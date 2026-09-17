#!/usr/bin/env bash
# VisionArena-only DFlash recipe. Run from the repository root.
# Usage and dependencies: docs/user_guide/tutorials/train_vlm_dflash.md
set -euo pipefail
if [[ -n "${DATASETS:-}${DATASET_PROPORTIONS:-}" || "${DATASET:-visionarena}" != visionarena ]]; then
    echo 'This launcher supports VisionArena only; remove DATASET/DATASETS/DATASET_PROPORTIONS overrides.' >&2
    exit 1
fi
VLM_SOURCE=visionarena
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen2_5_vl_7b_visionarena}"
source "$(dirname "${BASH_SOURCE[0]}")/vlm_dflash_common.sh"
