#!/usr/bin/env bash
# DFlash for a Qwen2.5-VL SFT target, using customer chat request bodies.
# The public model is a placeholder; MODEL must name the customer's actual
# checkpoint for both response regeneration and hidden-state extraction.
# Usage: REQUESTS_FILE=/data/requests.jsonl bash examples/train/dflash_qwen2_5_vl_7b_requests_online.sh
# Example request bodies: examples/data/requests.jsonl
# See docs/user_guide/tutorials/train_vlm_dflash.md for the input contract.
set -euo pipefail
: "${REQUESTS_FILE:?Set REQUESTS_FILE to a JSON/JSONL file of chat request bodies}"
VLM_SOURCE=requests
OUTPUT_DIR="${OUTPUT_DIR:-./output/dflash_qwen2_5_vl_7b_requests}"
# Draft architecture defaults; override through the environment as needed.
export NUM_LAYERS="${NUM_LAYERS:-5}"
export BLOCK_SIZE="${BLOCK_SIZE:-5}"
# Full vocabulary for Qwen/Qwen2.5-VL-7B-Instruct. When changing MODEL,
# set this to your target's config.vocab_size (or text_config.vocab_size).
export DRAFT_VOCAB_SIZE="${DRAFT_VOCAB_SIZE:-152064}"
source "$(dirname "${BASH_SOURCE[0]}")/vlm_dflash_common.sh"
