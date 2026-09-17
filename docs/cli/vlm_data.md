# Multimodal (VLM) data preparation

Use the [customer training guide](../user_guide/tutorials/train_vlm_dflash.md) for installation, VisionArena and customer request recipes, smoke tests, and resuming runs.

| Script                        | Purpose                                                        |
| ----------------------------- | -------------------------------------------------------------- |
| `export_visionarena.py`       | Export VisionArena prompts and images from a local cache       |
| `export_vlm_requests.py`      | Convert JSON/JSONL chat request bodies and base64/local images |
| `regenerate_vlm_responses.py` | Regenerate assistant turns using the target model              |

Both exporters produce prompt-only conversations consumed by regeneration, followed by `prepare_data.py` and online DFlash training. The VisionArena launcher uses VisionArena exclusively.

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
- `--fraction` - deterministic fraction of rows to keep after language filtering. It uses a nested hash threshold; raising the fraction preserves every previously selected row. It can be combined with `--limit`, which caps the selected rows.
- `--dataset-path` - directory holding the downloaded dataset: parquet shards (searched recursively) or a `save_to_disk` directory. Defaults to the cached snapshot in the HuggingFace cache (`$HF_HOME`, else `~/.cache/huggingface`). A cache entry holding only `README.md` is rejected rather than treated as an empty dataset.
- `--allow-download` - stream from the Hub instead, downloading the shards the run reaches. Off by default so a run cannot silently pull tens of GB.
- `--image-dir` - where image bytes are written. Filenames are content hashes, so reruns and images shared between conversations cost nothing. Pass this directory to vLLM as `--allowed-local-media-path`.
- `--max-turns` - cap user turns per conversation, which bounds how many generations regeneration needs.
- `--language` - keep only one language (e.g. `English`). Roughly half the dataset is English, so leaving this unset roughly doubles both the conversation count and the image bytes on disk.
- `--require-images` - skip text-only conversations. Off by default, since a VLM drafter also has to predict text-only turns.
- `--shuffle-buffer-size` - reservoir size for shuffling the stream. Rows hold image bytes, so this trades memory for mixing; shard order is shuffled regardless.

## export_vlm_requests.py

Converts customer Chat Completions request bodies from JSON/JSONL into the same prompt format. Base64 and local images are validated, deduplicated, and copied under `--image-dir`. System prompts and per-record generation parameters are preserved. See the [request format and examples](../user_guide/tutorials/train_vlm_dflash.md#customer-request-bodies).

```bash
python scripts/export_vlm_requests.py \
  --input requests.jsonl \
  --image-dir ./output/customer/images \
  --outfile ./output/customer/prompts.jsonl
```

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
- `--sampling-params` - JSON overrides merged over per-record `generation_params`. If neither specifies a setting, vLLM uses the model's `generation_config` defaults. Per-record `max_tokens` is capped by `--max-tokens`.
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
