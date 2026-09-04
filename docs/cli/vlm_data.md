# Multimodal (VLM) data preparation

These scripts build on-policy training data for a vision-language target model:

| Script                        | Purpose                                                                       |
| ----------------------------- | ----------------------------------------------------------------------------- |
| `export_visionarena.py`       | Turn a local VisionArena-Chat copy into prompt-only conversations plus images |
| `export_nemotron_vlm.py`      | Same, for Llama-Nemotron-VLM-Dataset-v1, selected by partition and fraction   |
| `download_nemotron_images.py` | Fetches images for the Nemotron partitions that ship without them             |
| `regenerate_vlm_responses.py` | Regenerate assistant responses with the target model, on-policy               |

The two exports emit the same prompt-only conversations, so everything downstream is shared: regeneration, then [`prepare_data.py`](prepare_data.md) like any other conversations JSONL. See `examples/train/dflash_qwen2_5_vl_7b_visionarena_online.sh` for a full recipe, where `DATASET=visionarena|nemotron` picks between them.

## Why multimodal needs its own path

[Response Regeneration](response_regeneration.md) writes speculator-format `input_ids`/`loss_mask` rows. That format cannot carry images: `prepare_data.py` passes pretokenized rows straight through and never adds the `messages` column that multimodal hidden-state extraction reads, so training would send only token ids to the Completions API. The images would be dropped and the image placeholder tokens embedded as ordinary text, producing hidden states for a model that never saw the picture.

`regenerate_vlm_responses.py` therefore emits natural-language `conversations` and leaves tokenization and loss masking to `prepare_data.py`, whose render endpoint already knows how images expand under the chat template.

Images are referenced by **path**, never inlined. `prepare_data.py` rejects base64 and decoded-image content parts so preprocessed datasets never copy image data, and every vLLM server that touches them needs `--allowed-local-media-path`.

## export_visionarena.py

`lmarena-ai/VisionArena-Chat` pairs ~199k real user vision prompts with responses from arena models (GPT-4o, Claude, and others). Those responses are off-policy for any target model you want to accelerate, so this script keeps the prompts and images and drops the responses.

It also absorbs two dataset quirks: `conversation` nests every turn in a single-element list, and `images` carries encoded bytes inline rather than paths.

The dataset is expected to be downloaded already. Shards are read from disk and nothing is fetched over the network, so the script works on an offline node:

```bash
hf download lmarena-ai/VisionArena-Chat --repo-type dataset
```

```bash
python scripts/export_visionarena.py \
  --limit 5000 \
  --max-turns 2 \
  --image-dir ./output/visionarena/images \
  --outfile ./output/visionarena/prompts.jsonl
```

Output rows are prompt-only, so they are an intermediate artifact: `prepare_data.py` cannot consume them until responses exist.

```json
{"conversation_id": "ab12", "conversations": [{"role": "user", "content": [
  {"type": "image", "path": "/abs/path/ab12.png"},
  {"type": "text", "text": "What's this?"}]}]}
```

**Parameters:**

- `--limit` - target number of conversations to use out of the ~199k. Rows are read as a stream, so a small limit touches only the shards it reaches rather than loading all ~84GB. With `--resume`, existing rows count toward the limit, so rerunning tops the file up instead of appending a second batch.
- `--dataset-path` - directory holding the downloaded dataset: parquet shards (searched recursively) or a `save_to_disk` directory. Defaults to the cached snapshot in the HuggingFace cache (`$HF_HOME`, else `~/.cache/huggingface`). A cache entry holding only `README.md` is rejected rather than treated as an empty dataset.
- `--allow-download` - stream from the Hub instead, downloading the shards the run reaches. Off by default so a run cannot silently pull tens of GB.
- `--image-dir` - where image bytes are written. Filenames are content hashes, so reruns and images shared between conversations cost nothing. Pass this directory to vLLM as `--allowed-local-media-path`.
- `--max-turns` - cap user turns per conversation, which bounds how many generations regeneration needs.
- `--language` - keep only one language (e.g. `English`). Roughly half the dataset is English, so leaving this unset roughly doubles both the conversation count and the image bytes on disk.
- `--require-images` - skip text-only conversations. Off by default, since a VLM drafter also has to predict text-only turns.
- `--shuffle-buffer-size` - reservoir size for shuffling the stream. Rows hold image bytes, so this trades memory for mixing; shard order is shuffled regardless.

## export_nemotron_vlm.py

`nvidia/Llama-Nemotron-VLM-Dataset-v1` holds 2.86M single-turn OCR, VQA and captioning rows over documents and charts, split into 21 partitions (`ocr_1`..`ocr_10`, `vqa_1`..`vqa_9`, `captioning_1`..`captioning_2`). Its assistant turns come from other models, so as with VisionArena this script keeps the prompts and images and drops the responses.

Three differences from VisionArena shape the script. Partitions are JSONL files with rows of the form `{"id", "image", "conversations": [{"from": "human"|"gpt", "value"}]}`, where the user turn carries an `<image>` placeholder marking the image's position in the text. Images live in webdataset TAR shards under `<partition>_images/`, keyed by member name, and the `image` field is that member name verbatim — so the tars are read directly and megatron-energon is not needed.

Most importantly, **the repo ships every partition's JSONL but only some partitions' images.** The rest are the entries commented out in `metadataset.yaml`; their images belong to the datasets the rows were annotated from and have to be fetched separately. Check what is usable before choosing:

```bash
python scripts/export_nemotron_vlm.py --list-partitions
```

A partition without local images is rejected rather than exported, since it would otherwise produce prompts referencing files that do not exist and fail much later, one vLLM request at a time. Five of the imageless partitions draw on OpenImages and can be fetched with [`download_nemotron_images.py`](#download_nemotron_imagespy); the error names them when they apply.

```bash
python scripts/export_nemotron_vlm.py \
  --partitions ocr_1,ocr_4,vqa_9 \
  --fraction 0.1 \
  --image-dir ./output/nemotron/images \
  --outfile ./output/nemotron/prompts.jsonl
```

**Parameters:**

- `--partitions` - comma-separated partitions to export. Default: every partition whose images are present. Order is preserved.
- `--fraction` - fraction of *each* partition to keep: `1` = 100%, `0.5` = 50%, `0.01` = 1%. This is the size knob, in place of an absolute row count.
- `--list-partitions` - print every partition with its row count, whether its images are present, and their size, then exit.
- `--dataset-path` - directory holding the downloaded dataset. Defaults to the cached snapshot in the HuggingFace cache. The dataset itself is never downloaded; unlike VisionArena there is no streaming option, because the images come from TAR shards that have to be on disk.
- `--image-source` - additional directory to look for `<partition>_images/` in, for images fetched by `download_nemotron_images.py`. It *adds* a location rather than replacing `--dataset-path`, so the shipped TAR shards stay visible alongside downloaded images.
- `--image-dir` - where images are extracted, one subdirectory per partition. Nested member names are flattened (`data/train/x/1.jpg` becomes `data__train__x__1.jpg`) so that a name from the archive cannot decide where the file lands. Downloaded images are *not* copied here; see below.
- `--seed` - salts the selection hash. Changing it selects a different subset of the same size, so leave it alone when topping up.

Images that came from TAR shards are extracted under `--image-dir`, but downloaded ones are referenced where they already sit — copying them would be a second copy of a selection that reaches 378 GB for `vqa_1` alone. The export logs every root images are referenced from, and warns if there is more than one, because vLLM accepts a single `--allowed-local-media-path`. Downloading with `--image-source` set to the export's `--image-dir` keeps everything under one root, which is what the training script does.

**The images have to stay on disk for the whole run, including training.** `prepare_data.py` stores a `messages` column holding `file://` URLs rather than pixels, and online hidden-state extraction sends those messages to vLLM, which reads the files at that moment. Deleting the images after preprocessing does not free space early — it breaks training. Budget for them alongside the prepared dataset, not instead of it.

### Why fraction sampling is a hash threshold

Selection compares a digest of each row id against `--fraction` rather than shuffling and slicing. That makes it deterministic without storing any state, and **nested**: every row kept at `0.1` is also kept at `0.2`. Raising the fraction therefore keeps everything already exported and only adds to it, so scaling up reuses the images already extracted. A shuffle-and-slice sample would pick a different subset at the new size and re-extract its images, throwing away the previous run's work.

That property is also what lets the downloader fetch only what will be exported: it calls the same `keeps_row`, so the same `--fraction` and `--seed` name the same subset in both scripts.

## download_nemotron_images.py

Fetches images for the Nemotron partitions that ship without them. Only the five whose images are addressable from the row's `image` field alone can be fetched unattended — `captioning_1`, `captioning_2`, `vqa_1`, `vqa_2`, `vqa_3`, all on OpenImages at `https://s3.amazonaws.com/open-images-dataset/train/{image}`. The others (ChartQA, DocLayNet, PubTables-1M, TextVQA archives) need their source dataset obtained by hand, and are rejected with a pointer to the partition's `.md`.

**The download is sampled with the export.** `vqa_1` is 1,278,221 images and about 378 GB at full size, and a run training on a tenth of it has no use for the other 340 GB. Passing the same `--fraction` and `--seed` as the export fetches exactly the images the export will ask for.

```bash
# size it first
python scripts/download_nemotron_images.py \
  --partitions vqa_1 --fraction 0.05 --dry-run

# fetch, then export the same selection
python scripts/download_nemotron_images.py \
  --partitions vqa_1 --fraction 0.05 --image-source ./output/nemotron/images
python scripts/export_nemotron_vlm.py \
  --partitions vqa_1 --fraction 0.05 --image-source ./output/nemotron/images \
  --image-dir ./output/nemotron/images --outfile ./output/nemotron/prompts.jsonl
```

**Parameters:**

- `--partitions` - comma-separated partitions to fetch images for. Required.
- `--fraction`, `--seed` - the sampling to fetch for. Must match the export's, or the two disagree about which rows are in play.
- `--image-source` - directory to create `<partition>_images/` under. Defaults to the dataset directory, which is where the export looks with no extra flags; point it at the export's `--image-dir` to keep one vLLM media root.
- `--dataset-path` - directory holding the partition JSONLs. Defaults to the HuggingFace cache copy.
- `--concurrency` - simultaneous requests, default 64. Around 440 images/s (150 MB/s) was measured at that setting and 780 images/s (220 MB/s) at 128, so full `vqa_1` takes roughly half an hour.
- `--max-retries`, `--timeout` - per-image attempts and per-request timeout.
- `--dry-run` - report how many images are missing and their approximate size, then exit.

Downloads are resumable: an image already on disk is never refetched, so an interrupted run is continued by rerunning the same command, and raising `--fraction` fetches only the newly selected images. Each file is written under a temporary name and renamed, so an interrupted run cannot leave a truncated image that later looks complete.

Expect a small number of permanent failures. OpenImages has removed keys over the years — about 3.4% of `vqa_1` returns 404 — so the usable row count is a few percent below nominal. These are reported rather than retried, and the export counts the rows it had to drop.

## regenerate_vlm_responses.py

Generates the target model's own responses for the exported prompts. Multi-turn conversations are regenerated sequentially, so each turn conditions on the model's own prior responses rather than the arena's, keeping the whole assistant history on-policy.

Run it against a **plain** vLLM server, not one launched by [`launch_vllm.py`](launch_vllm.md): the hidden-states server writes a safetensors file for every request it serves, so multi-token generation against it would fill the disk.

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --allowed-local-media-path "$(realpath ./output/visionarena/images)" &

python scripts/regenerate_vlm_responses.py \
  --data ./output/visionarena/prompts.jsonl \
  --outfile ./output/visionarena/conversations.jsonl \
  --endpoint http://localhost:8000/v1/chat/completions
```

**Parameters:**

- `--data` / `--outfile` - input prompt-only JSONL and output conversations JSONL.
- `--endpoint` - full Chat Completions path, unlike `prepare_data.py`'s `--render-endpoint`, which takes a base URL.
- `--model` - served model id; auto-detected from `/v1/models` if omitted.
- `--max-tokens` - cap per generated turn. A response that hits the cap ends regeneration for that conversation rather than conditioning later turns on a cut-off answer.
- `--sampling-params` - JSON merged into each request. Left empty, vLLM applies the model's own `generation_config` defaults, which is what serving would use and therefore what "on-policy" means here.
- `--concurrency`, `--max-retries`, `--limit`, `--resume`.

Failed conversations go to a sibling `.errors.jsonl` file rather than the training input, and `--resume` skips conversations already in the output.

## Feeding prepare_data.py

The regenerated conversations are ordinary input, but the render endpoint is required and must be a server that can read the images:

```bash
python scripts/prepare_data.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --data ./output/visionarena/conversations.jsonl \
  --output ./output/visionarena/prepared \
  --render-endpoint http://localhost:8000 \
  --seq-length 8192
```

The resulting dataset carries `input_ids`, `loss_mask`, `seq_len`, and `messages`. That last column is what keeps the images reachable: training reads it to route the row to the Chat Completions API for hidden-state extraction. Its absence is the silent failure this whole path exists to avoid, so it is worth checking once on a new target model:

```python
from datasets import load_from_disk
assert "messages" in load_from_disk("./output/visionarena/prepared").column_names
```

Bound how many tokens each image expands to when serving. Qwen2.5-VL's default (`max_pixels=12845056`) lets one large image consume most of the context:

```bash
--mm-processor-kwargs '{"max_pixels": 1003520}' --limit-mm-per-prompt '{"image": 4}'
```

Use the same value on the regeneration and hidden-states servers, so responses are generated at the resolution the drafter is trained on.
