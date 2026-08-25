#!/usr/bin/env python3
"""
Export VisionArena-Chat prompts and images for on-policy VLM data generation.

``lmarena-ai/VisionArena-Chat`` pairs real user vision prompts with responses
from arena models (GPT-4o, Claude, and others), so its assistant turns are
off-policy for whatever target model you want to accelerate. This script keeps
the part that transfers -- the user prompts and their images -- and drops the
responses. ``regenerate_vlm_responses.py`` then regenerates them with the target
model, and ``prepare_data.py`` converts the result into training rows.

Two dataset quirks are handled here:

1. ``conversation`` nests every turn in a single-element list, so turns are
   flattened before use.
2. ``images`` carries encoded bytes inline. ``prepare_data.py`` rejects inline
   and base64 images so that preprocessed datasets never copy them, so the
   bytes are written to ``--image-dir`` and referenced by path instead. The
   dataset names images by content hash, which makes the export idempotent and
   deduplicates images shared across conversations.

Images are attached to the first user turn, matching how the arena presents a
single upload to the whole conversation.

The output is a JSONL file of prompt-only conversations:

    {"conversation_id": "ab12...", "conversations": [
        {"role": "user", "content": [
            {"type": "image", "path": "/abs/path/ab12.png"},
            {"type": "text", "text": "What's this?"}]}]}

These rows are an intermediate artifact: they have no assistant turns, so
``prepare_data.py`` cannot consume them until responses are generated.

The dataset is ~199k rows / ~84GB, so it is read as a stream and ``--limit``
stops early rather than downloading every shard first.

Usage:
    python scripts/export_visionarena.py \
        --limit 5000 \
        --image-dir ./output/visionarena/images \
        --outfile ./output/visionarena/prompts.jsonl
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Streaming a shuffled dataset issues a request per row group across every
# shard, and httpx logs each one at INFO. Left alone that buries the export's
# own progress and writes one line per request -- gigabytes at a 200k limit --
# each containing a presigned CDN URL carrying the account id and signature.
for _noisy in ("httpx", "httpcore", "urllib3", "hf_xet", "filelock", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

VISIONARENA_HF_PATH = "lmarena-ai/VisionArena-Chat"
VISIONARENA_SPLIT = "train"

# Leading bytes are enough to name a file usefully; only used when a row is
# missing the dataset's own content-hash filename.
_MAGIC_SUFFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export VisionArena-Chat user prompts and images as prompt-only "
            "conversations for on-policy response regeneration."
        )
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("./output/visionarena/prompts.jsonl"),
        help=(
            "JSONL path for the exported prompt-only conversations "
            "(default: ./output/visionarena/prompts.jsonl)"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write image files into (default: 'images' next to "
            "--outfile). Serve vLLM with --allowed-local-media-path pointing here."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Stop once the output file holds this many conversations (default: "
            "all ~199k). With --resume, rows already in the file count toward "
            "it, so rerunning the same command tops the file up to --limit "
            "rather than growing it by --limit again."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Keep only rows whose 'language' column equals this value "
            "(e.g. English). Default: keep all languages."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help=(
            "Keep at most this many user turns per conversation. Later turns are "
            "dropped, which also bounds the number of generations regeneration "
            "needs. Default: keep all turns."
        ),
    )
    parser.add_argument(
        "--require-images",
        action="store_true",
        help=(
            "Skip conversations that carry no usable image. By default text-only "
            "rows are kept, since a VLM drafter also has to predict text-only turns."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to --outfile, skipping conversations it already contains",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed for the dataset stream (default: 0)",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=500,
        help=(
            "Reservoir size for shuffling the stream. Rows hold encoded image "
            "bytes (~1MB each), so this buys mixing with memory: 500 costs "
            "roughly 0.5GB. Shard order is shuffled regardless, which does most "
            "of the work. Set to 0 to read shards in order (default: 500)."
        ),
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    if args.max_turns is not None and args.max_turns <= 0:
        parser.error("--max-turns must be > 0")
    if args.shuffle_buffer_size < 0:
        parser.error("--shuffle-buffer-size must be >= 0")
    if args.image_dir is None:
        args.image_dir = args.outfile.parent / "images"
    return args


def flatten_conversation(conversation: Any) -> list[dict]:
    """Flatten VisionArena's ``List(List(turn))`` column into a list of turns.

    Every turn arrives wrapped in a single-element list. Bare dicts are also
    accepted so the function tolerates the already-flat shape.
    """
    if not isinstance(conversation, list):
        return []

    turns: list[dict] = []
    for entry in conversation:
        if isinstance(entry, dict):
            turns.append(entry)
        elif isinstance(entry, list):
            turns.extend(turn for turn in entry if isinstance(turn, dict))
    return turns


def extract_prompt_turns(turns: list[dict], max_turns: int | None = None) -> list[dict]:
    """Keep the turns that drive regeneration, dropping the arena's responses.

    Assistant turns are off-policy and are regenerated by the target model, so
    only system and user turns survive. ``max_turns`` caps user turns; a trailing
    system turn is dropped with them since nothing would follow it.
    """
    prompt_turns: list[dict] = []
    num_user_turns = 0

    for turn in turns:
        role = turn.get("role") or turn.get("from")
        content = turn.get("content")
        if content is None:
            content = turn.get("value")

        if role in ("user", "human"):
            if max_turns is not None and num_user_turns >= max_turns:
                break
            if not content:
                # An empty user turn gives the model nothing to answer, and
                # keeping it would shift every later turn onto the wrong image.
                return []
            prompt_turns.append({"role": "user", "content": content})
            num_user_turns += 1
        elif role == "system" and content:
            prompt_turns.append({"role": "system", "content": content})

    if num_user_turns == 0:
        return []
    # Trailing system turns would leave the conversation without a final
    # question for the model to answer.
    while prompt_turns and prompt_turns[-1]["role"] == "system":
        prompt_turns.pop()
    return prompt_turns


def _guess_suffix(data: bytes) -> str:
    for magic, suffix in _MAGIC_SUFFIXES:
        if data.startswith(magic):
            return suffix
    if data[8:12] == b"WEBP":
        return ".webp"
    return ".img"


def _image_filename(image: dict, data: bytes) -> str:
    """Filename for one image, preferring the dataset's content-hash name.

    Only the basename is used: the value comes from the dataset and must not be
    able to write outside ``--image-dir``.
    """
    name = Path(str(image.get("path") or "")).name
    if name:
        return name
    return hashlib.sha256(data).hexdigest()[:32] + _guess_suffix(data)


def write_images(images: Any, image_dir: Path) -> list[Path]:
    """Write a row's inline image bytes into ``image_dir``; return their paths.

    Names are content hashes, so an image already on disk is left alone: reruns
    and images shared between conversations cost nothing.
    """
    if not isinstance(images, list):
        return []

    paths: list[Path] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        data = image.get("bytes")
        if not data:
            # A decoded PIL image or a bare remote path has no bytes to write.
            continue
        destination = image_dir / _image_filename(image, data)
        if not destination.exists():
            destination.write_bytes(data)
        paths.append(destination)
    return paths


def attach_images(turns: list[dict], image_paths: list[Path]) -> list[dict]:
    """Attach the row's images to its first user turn as content parts.

    VisionArena scopes images to the conversation rather than a turn, matching
    an arena upload that stays visible for the whole chat. Text-only rows keep
    their plain-string content, which the render path passes through unchanged.
    """
    if not image_paths:
        return turns

    attached = False
    result: list[dict] = []
    for turn in turns:
        if attached or turn["role"] != "user":
            result.append(turn)
            continue

        parts: list[dict] = [
            {"type": "image", "path": str(path.resolve())} for path in image_paths
        ]
        parts.append({"type": "text", "text": turn["content"]})
        result.append({"role": "user", "content": parts})
        attached = True
    return result


def load_exported_ids(path: Path) -> set[str]:
    """Conversation ids already present in an output file, for ``--resume``."""
    exported: set[str] = set()
    if not path.is_file():
        return exported

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversation_id = row.get("conversation_id")
            if conversation_id:
                exported.add(str(conversation_id))
    return exported


def build_row(
    row: dict, image_dir: Path, *, max_turns: int | None, require_images: bool
) -> dict | None:
    """Convert one dataset row into a prompt-only conversation, or skip it."""
    turns = extract_prompt_turns(
        flatten_conversation(row.get("conversation")), max_turns
    )
    if not turns:
        return None

    image_paths = write_images(row.get("images"), image_dir)
    if require_images and not image_paths:
        return None

    return {
        "conversation_id": row.get("conversation_id"),
        "conversations": attach_images(turns, image_paths),
    }


def iter_dataset(seed: int, shuffle_buffer_size: int) -> Iterator[dict]:
    """Stream VisionArena-Chat, optionally shuffling within a reservoir.

    Streaming keeps the ~84GB of image shards from being downloaded up front;
    only the shards a run actually reaches are fetched.
    """
    dataset = load_dataset(VISIONARENA_HF_PATH, split=VISIONARENA_SPLIT, streaming=True)
    if shuffle_buffer_size:
        # Shards are ordered by upload time, so an unshuffled prefix is skewed
        # toward whichever models and prompt styles were live then. This also
        # shuffles shard order, so the buffer stays small enough that holding
        # image bytes in it is affordable.
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)
    return iter(dataset)


def main() -> None:
    args = parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    exported_ids = load_exported_ids(args.outfile) if args.resume else set()
    if exported_ids:
        logger.info(
            "Resuming: %d conversations already in %s", len(exported_ids), args.outfile
        )

    logger.info("Streaming %s (split %s)", VISIONARENA_HF_PATH, VISIONARENA_SPLIT)
    logger.info("Writing images to %s", args.image_dir.resolve())
    logger.info(
        "Serve vLLM with --allowed-local-media-path %s so it can read them",
        args.image_dir.resolve(),
    )

    # --limit is a target size for the file, not a per-run quota, so a rerun
    # tops it up instead of appending a second batch.
    num_existing = len(exported_ids)
    num_exported = 0
    num_skipped = 0

    if args.limit is not None and num_existing >= args.limit:
        logger.info(
            "Output file already holds %d conversations; nothing to do", num_existing
        )
        return

    with (
        args.outfile.open("a" if args.resume else "w", encoding="utf-8") as handle,
        tqdm(
            total=args.limit, initial=num_existing, desc="Exporting", unit="conv"
        ) as progress,
    ):
        for row in iter_dataset(args.seed, args.shuffle_buffer_size):
            if args.limit is not None and num_existing + num_exported >= args.limit:
                break
            if args.language is not None and row.get("language") != args.language:
                continue
            if row.get("conversation_id") in exported_ids:
                continue

            exported = build_row(
                row,
                args.image_dir,
                max_turns=args.max_turns,
                require_images=args.require_images,
            )
            if exported is None:
                num_skipped += 1
                continue

            handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
            num_exported += 1
            progress.update(1)
        handle.flush()

    logger.info(
        "Exported %d conversations to %s (%d now in file, %d rows skipped as unusable)",
        num_exported,
        args.outfile,
        num_existing + num_exported,
        num_skipped,
    )
    logger.info(
        "Next: generate on-policy responses with scripts/regenerate_vlm_responses.py"
    )


if __name__ == "__main__":
    main()
    # Leaving a partially-consumed streaming dataset behind -- which --limit
    # always does -- leaves datasets' parquet reader with a live native thread
    # that aborts during interpreter finalization (SIGABRT, exit 134), long
    # after the export itself succeeded. Reproducible with nothing but
    # load_dataset(streaming=True) and a break, and neither closing the
    # iterator nor gc.collect() prevents it. Exit before finalization runs;
    # the export is already written and flushed above. Without this the script
    # reports failure on every successful run, which would stop any pipeline
    # that checks its exit code.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
