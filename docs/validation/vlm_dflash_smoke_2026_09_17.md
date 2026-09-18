# VLM DFlash smoke validation — 2026-09-17

The VisionArena and customer-request launchers were exercised against the public `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint on NVIDIA H200 GPUs. This checks pipeline execution, image preservation, training, and checkpoint creation; it is not a model-quality or speculative-decoding benchmark.

Tested source: [`cc003cdeaf41b451122a3e4582d7f766b0956fec`](https://github.com/johncalesp/speculators/commit/cc003cdeaf41b451122a3e4582d7f766b0956fec), cloned from `jcalderon/vlm-experimentation`. The follow-up documentation commit adds this report and its artifacts without changing the tested implementation.

## Scope

Each full run used 12 records, one training epoch, sequence length 2,048, a 128-token response cap, one draft layer, block size 4, 32 anchors, and one GPU for each of the extraction server and trainer. VisionArena used cached English conversations with one user turn. Customer records used the corrected synthetic attribute-extraction request, with twelve distinct IDs sharing one image.

Both pipelines performed:

1. Prompt export and image materialization.
2. Response generation by the target, with zero request failures.
3. Rendering/preprocessing with multimodal `messages` and nonempty assistant loss masks.
4. Online hidden-state extraction and DFlash optimization/validation.
5. Checkpoint writing, including source SHA and package versions in `train_command.txt`.

Additional checks covered resume into a second epoch with regeneration and preprocessing reused, refusal to reuse data after generation settings changed, and 71 local tests across VLM preprocessing and training-data handling. The tests include streaming 2,000 customer records, image deduplication, repeat export, invalid-image handling, atomic export failure, and sampling-parameter propagation. Python lint/format checks, Bash syntax checks, and `git diff --check` passed.

## Environment and reproducibility

The environment used Python 3.12 in `vllm/vllm-openai:v0.25.1`, with this repository installed into a virtual environment using the image's system packages. Git was made available inside the container for provenance capture. All generated data, code, caches, and logs stayed beneath the authorized remote workspace; the existing Hugging Face cache was mounted read-only.

- Target snapshot: `cc594898137f460bfe9f0759e9844b3ce807cfb5`.
- VisionArena snapshot: `1394b4f59ab6f1f2e5aff6bc15b448e15960e170`.
- [Exact launcher environment and commands](vlm_dflash_smoke_2026_09_17/launch_smoke.sh).
- [Machine-readable validation and package versions](vlm_dflash_smoke_2026_09_17/validation.json).
- VisionArena: [training provenance](vlm_dflash_smoke_2026_09_17/visionarena_train_command.txt), [pipeline configuration](vlm_dflash_smoke_2026_09_17/visionarena_pipeline_config.json).
- Customer requests: [training provenance](vlm_dflash_smoke_2026_09_17/requests_train_command.txt), [pipeline configuration](vlm_dflash_smoke_2026_09_17/requests_pipeline_config.json).

The launcher artifact uses `/workspace` as a generic container mount point for the working directory. Checkpoints, images, prepared rows, and logs remain there in `visionarena-smoke-final` and `requests-smoke-final`. They are not committed to the repository. Use the [customer guide](../user_guide/tutorials/train_vlm_dflash.md) for portable setup and commands.

## Limits

Only the small, one-trainer-GPU configuration was exercised on hardware. The four-GPU defaults and full-scale training were not benchmarked. The short output budget truncates some VisionArena responses; exact counts are in the validation artifact. The repeated synthetic customer image is unsuitable for measuring training quality or generalization. The private SFT checkpoint was unavailable, so it has not been tested.

The base container's `pip check` reports pre-existing PyGObject/pycairo and NIXL version inconsistencies. The tested pipelines use the file hidden-state connector and completed successfully; those optional integrations were not exercised.
