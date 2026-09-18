# VLM DFlash validation with vLLM 0.29.0

Both launchers completed end-to-end smoke runs on NVIDIA H200 GPUs with vLLM **0.29.0** and the new full-vocabulary default, **`DRAFT_VOCAB_SIZE=152064`**. The target's cached `config.json` confirms `vocab_size=152064` for `Qwen/Qwen2.5-VL-7B-Instruct`.

Tested source: [`f5dfd20f65ad906b20981d24c58a4c2d2878e18a`](https://github.com/johncalesp/speculators/commit/f5dfd20f65ad906b20981d24c58a4c2d2878e18a), checked out from `jcalderon/vlm-experimentation` on 2026-09-17.

## Checks performed

| Pipeline          | Exported | Regenerated     | Prepared           | Training                             |
| ----------------- | -------- | --------------- | ------------------ | ------------------------------------ |
| VisionArena       | 12       | 12, zero errors | 12 multimodal rows | One epoch plus validation/checkpoint |
| Customer requests | 12       | 12, zero errors | 12 multimodal rows | One epoch plus validation/checkpoint |

The runs cover image export, response regeneration, rendering, online hidden-state extraction, DFlash optimization, validation, and checkpoint writing. Both use sequence length 2,048, a 128-token generation cap, a one-layer drafter, block size 4, 32 anchors, and one extraction GPU plus one training GPU. Customer records are synthetic copies of the example with distinct IDs. `DRAFT_VOCAB_SIZE` was left unset in the parent environment to exercise the launchers' default.

For each checkpoint, validation confirms `draft_vocab_size=152064` in `config.json`, `lm_head.weight` shape **`[152064, 3584]`**, and `--draft-vocab-size 152064` in `train_command.txt`. Every prepared row retains image messages and a nonempty assistant loss mask. Both processes exit with code zero. The 71 VLM preprocessing and training-data unit tests also pass inside this vLLM 0.29.0 environment, and `python -m pip check` reports no broken requirements:

```bash
python -m pytest tests/unit/scripts/test_vlm_data_prep.py tests/unit/train/test_data.py -q --tb=short
```

## Reproducible environment

- Container: `vllm/vllm-openai:v0.29.0`.
- Image digest: `sha256:c2914767605584b6d8f45686b82de173ecc99e781897aa3d0a66dacd72c51ae1`.
- Python 3.12, PyTorch `2.13.0+cu130`, Transformers `5.15.1`.
- Target snapshot: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- VisionArena snapshot: `1394b4f59ab6f1f2e5aff6bc15b448e15960e170`.

The image bundles Transformers `5.16.1`, which is outside this repository's `<5.16.0` requirement. A separate virtual environment using system packages installs this repository and Transformers `5.15.1`; that version also satisfies vLLM's `>=5.10.4` requirement. vLLM remains exactly `0.29.0`, and the image's PyTorch version is retained. Git is available inside the container for source provenance. The shared Hugging Face cache is mounted read-only, and all test outputs are inside the authorized workspace.

Artifacts:

- [Exact launch commands and environment](vlm_dflash_vllm_0_29_0/launch_smoke.sh).
- [Validation results and package versions](vlm_dflash_vllm_0_29_0/validation.json).
- VisionArena: [training provenance](vlm_dflash_vllm_0_29_0/visionarena_train_command.txt), [pipeline configuration](vlm_dflash_vllm_0_29_0/visionarena_pipeline_config.json).
- Customer requests: [training provenance](vlm_dflash_vllm_0_29_0/requests_train_command.txt), [pipeline configuration](vlm_dflash_vllm_0_29_0/requests_pipeline_config.json).

The working directory is mounted at `/workspace` inside the test container. Checkpoints and datasets remain in `visionarena-smoke-vllm029` and `requests-smoke-vllm029`; model weights are not committed to Git. See the [customer guide](../user_guide/tutorials/train_vlm_dflash.md) for portable setup and usage.

## Limits

These are execution checks, not training-quality or speculative-decoding benchmarks. The short generation limit truncates some VisionArena responses; counts are recorded in the validation artifact. Full-scale training, the default four-GPU allocation, and a private SFT target were not exercised in this run. Use a fresh output directory when migrating from the previous 32,000-token draft vocabulary because its mappings and checkpoints have different dimensions.
