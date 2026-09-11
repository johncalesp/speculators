# DFlash2 / Qwen2.5-VL-7B / Cauldron

Submit from the repository root on the Slurm login node:

```bash
sbatch slurm/training_script_cauldron.sh
```

Defaults: all 50 Cauldron subsets, all source conversations, five epochs.
Choose ten epochs **before the first submission**:

```bash
EPOCHS=10 sbatch slurm/training_script_cauldron.sh
```

For a separate small validation run:

```bash
OUTPUT_DIR=/workspace/sandbox/dflash2_training/output/cauldron_smoke \
CAULDRON_SUBSETS=ai2d MAX_SAMPLES=128 EPOCHS=1 \
sbatch slurm/training_script_cauldron.sh
```

The supplied partition, account, container image, mounts, and work directory
come from the original Slurm skeleton. Check those paths on the destination
cluster. Submit from the repo root, or set PIPELINE_SBATCH_SCRIPT to the script's
absolute **host** path. WORK_DIR and OUTPUT_DIR are **container** paths.

## Continuation across five-hour allocations

Each allocation requests four GPUs. The default work deadline is 4 hours
30 minutes after the batch shell starts, including container startup. This
leaves 30 minutes for a training step, checkpoint save, and process cleanup.
At that deadline the controller creates a stop file.

- Data preparation finishes in-flight conversations where possible. Each
  finished conversation is flushed to a persistent journal. A torn final
  append is discarded on restart. Completed Arrow chunks are reused.
- Training checks for a stop request after every optimizer/scheduler update.
  All ranks agree before pausing. It records the epoch, completed batch count,
  global step, model, optimizer, and scheduler. Resume skips completed batches.
- Checkpoints are also saved every 200 optimizer steps and at epoch boundaries.
  A replacement is written separately before publication. If a process dies
  during publication, the previous complete checkpoint can be recovered.
- Exit code 75 makes the batch script submit **one** successor with
  `--dependency=afterany:<current_job_id>`. The successor receives the same
  exported run settings and starts after the current allocation exits.
- Successful completion writes OUTPUT_DIR/DONE.json and stops the chain.
  Ordinary failures stop the chain too. Three consecutive allocations with no
  measurable progress also stop it.

This uses new dependent jobs, rather than requiring permission to requeue a
running job. The cluster must allow `sbatch` from compute nodes. The dependency
behavior is documented in [Slurm's sbatch reference](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

Set AUTO_CONTINUE=0 to save progress without submitting a successor. Submitting
the same script again resumes the same OUTPUT_DIR. Cancel the active job with
`scancel JOB_ID` to stop work; if a successor was already printed, cancel that
job as well. Hardware failures or a hard kill before the controller can return
do not automatically submit a successor; manually resubmit after inspecting
the logs.

The normal time-budget stop is coordinated. An unexpected training signal
retains the last committed checkpoint instead of trusting a partly completed
optimizer step. Standard Speculators checkpoint precision and random-seed
behavior still apply; resume is not a claim of bitwise-identical training.
A pause during validation resumes from the completed training epoch; the
unfinished validation pass is not repeated.

## Cached data and environment

The container's Python is used; the workstation's .venv is not activated.
The wrapper exposes this checkout's src and hs_connectors/src on PYTHONPATH.
The container must already contain the Speculators runtime dependencies
(including datasets, pyarrow, pydantic-settings, PyYAML), PyTorch, Transformers,
and a compatible vLLM with hidden-state extraction and the render endpoint.
Prepare the container/environment before submitting the long run. PYTHON can
select a different compatible interpreter inside the container.

HF_HOME defaults to /workspace/.cache. Hub and Datasets offline modes are
enabled: **both the complete target model and dataset must be cached**.
The default data source is a Hugging Face **Hub snapshot of the raw Parquet
repository**, as produced by:

```bash
# Run while staging data, before the offline Slurm workflow.
HF_HOME=/workspace/.cache hf download --repo-type dataset HuggingFaceM4/the_cauldron
HF_HOME=/workspace/.cache hf download Qwen/Qwen2.5-VL-7B-Instruct
```

A transformed Arrow cache made only by datasets.load_dataset is not the same
layout. Set CAULDRON_DATA_DIR to an existing raw dataset snapshot/local mirror
when it is not in the Hub cache. It must contain README.md and the subset
directories with their Parquet shards. Explicit CAULDRON_SUBSETS permits a
partial download containing those subsets; "all" checks the dataset card's
declared subset list and each subset's shard count.

OUTPUT_DIR must be on the persistent mount, at the same container path in
every allocation. It contains exported image files, conversations, regenerated
responses, tokenized Arrow data, and checkpoints. Budget disk space for exported
images in addition to the original dataset cache. The final prepared dataset
links to its immutable chunks, so keep the entire output tree at the same path.

## Data and training behavior

[Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) stores an
images list and a texts list containing user/assistant pairs. The exporter
retains the images and converts each row into a multi-turn conversation.
Original assistant answers are kept in the export for inspection; the target
model regenerates every assistant turn for training, conditioning later turns
on its own earlier answers.

Preparation uses all four GPUs for the target model with tensor parallelism 4.
It processes 128 source rows per chunk, interleaving subsets. Response
regeneration and rendering happen per conversation, with 16 concurrent
conversations by default. The repository's rendering-boundary implementation
produces assistant loss masks, and every training row retains its image
references. Completed chunks are assembled through Arrow file links without
recopying the full corpus.

Conversations exceeding MAX_IMAGES are reported as skipped. Later turns whose
context fills the window, or whose response hits the generation cap, are
reported and omitted; earlier complete turns can still be retained. Empty
responses, permanent HTTP errors, missing images, and malformed inputs fail the
job so a configuration/server error does not silently remove the dataset.
Per-chunk complete.json files report retained rows and skip reasons.

Training uses two GPUs for vLLM extraction (tensor parallelism 2) and two for
the DFlash2 trainer with FSDP and activation checkpointing. GPU indices are
mapped through the allocation's CUDA_VISIBLE_DEVICES. Hidden states are
generated online and deleted after consumption, avoiding a persistent hidden
state cache for the full corpus. The target's 28 text layers give auxiliary
layer IDs 2, 14, 25; the extraction launcher additionally includes final layer
28. The draft uses the full 152064-token vocabulary.

| Setting | Default |
| --- | --- |
| EPOCHS | 5 |
| CAULDRON_SUBSETS | all; comma- or space-separated names also accepted |
| MAX_SAMPLES | 0 = all source conversations |
| CHUNK_SIZE | 128 source conversations |
| SEQ_LENGTH | 8192 |
| MAX_NEW_TOKENS | 512 per assistant answer |
| TEMPERATURE | 0 |
| MAX_IMAGES | 16 per conversation |
| MAX_PIXELS | 1003520 per image |
| REGEN_CONCURRENCY | 16 |
| NUM_LAYERS / BLOCK_SIZE / MAX_ANCHORS | 5 / 8 / 512 |
| LR | 0.0003 |
| CHECKPOINT_STEPS | 200 |
| VLLM_PORT | 8000 |
| WORK_SECONDS | 16200 (4h30m) |
| SAVE_GRACE_SECONDS | 1200 (20 minutes) |

Do not increase WORK_SECONDS beyond 16200 or SAVE_GRACE_SECONDS beyond 1200
within a five-hour allocation. Server startup has a 1200-second timeout,
adjustable with SERVER_START_TIMEOUT.

The resolved pipeline configuration and source digest are frozen on first use.
Changing data, model snapshots, training settings, or source code requires a
new OUTPUT_DIR (or restoring the original settings). This prevents resuming a
checkpoint against different sample indices or a different learning-rate
schedule. This recipe targets Qwen/Qwen2.5-VL-7B-Instruct specifically.

DFlash2's objective in this branch remains experimental. Four-GPU VLM training
and the destination cluster/container need a smoke run; local tests do not
establish training quality or an ETA for the complete Cauldron corpus.

## Results and logs

- Slurm output: the normal slurm-JOB_ID.out file, including the successor job ID.
- OUTPUT_DIR/provenance/JOB_ID/: stage command lines, logs, source copies,
  vLLM provenance, checkpoint hashes, and patches for that allocation.
- OUTPUT_DIR/chunks/CHUNK_ID/: exported images and conversations,
  regenerated.jsonl (including prepared rows), Arrow shard, completion report.
- OUTPUT_DIR/prepared/: composed training dataset.
- OUTPUT_DIR/checkpoints/EPOCH/: checkpoints numbered from zero.
- OUTPUT_DIR/checkpoints/checkpoint_best: best fully evaluated checkpoint.
- OUTPUT_DIR/continuation.json: last saved progress and stalled-allocation count.
- OUTPUT_DIR/DONE.json: final epoch checkpoint when training finishes.

Keep the whole provenance tree with any published model/results, alongside
the checkpoint's train_command.txt, run.yaml, and patches. Every invocation
of scripts/launch_vllm.py passes --provenance-dir. Normal response generation
uses the same provenance writer without enabling hidden-state dumping.

## Local checks

Run with an environment containing this checkout's dependencies:

```bash
bash -n slurm/training_script_cauldron.sh slurm/run_cauldron.sh
PYTHONPATH=slurm:src:hs_connectors/src python -m pytest \
  -q -c /dev/null -p no:cacheprovider --confcutdir=slurm/tests slurm/tests
python -m ruff check slurm
python -m ruff format --check slurm
```
