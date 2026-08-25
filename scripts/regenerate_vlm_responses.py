#!/usr/bin/env python3
"""
Regenerate assistant responses for multimodal conversations with a target VLM.

Speculator training data has to contain the target model's own responses. For
text-only datasets ``scripts/response_regeneration/`` does this, but it emits
speculator-format ``input_ids``/``loss_mask`` rows, and that format cannot carry
images: ``prepare_data.py`` passes pretokenized rows straight through without
the ``messages`` column that multimodal hidden-state extraction needs, so the
images would be silently dropped and the image placeholder tokens embedded as
ordinary text. This script therefore emits natural-language ``conversations``
and leaves tokenization and loss masking to ``prepare_data.py``, whose render
endpoint already knows how images expand under the chat template.

Input is a JSONL file of prompt-only conversations, as written by
``scripts/export_visionarena.py``:

    {"conversations": [{"role": "user", "content": [
        {"type": "image", "path": "/abs/img.png"}, {"type": "text", "text": "..."}]}]}

Assistant turns in the input are ignored and regenerated. Multi-turn
conversations are regenerated sequentially so each turn conditions on the
model's own prior responses rather than the original dataset's, keeping the
whole assistant history on-policy.

Output rows carry the same conversation with assistant turns filled in, ready
for ``prepare_data.py --data <outfile>``. Image parts keep their
``{"type": "image", "path": ...}`` form, which is what ``prepare_data.py``
accepts; the ``image_url`` form used on the wire is not.

The vLLM server must be started with ``--allowed-local-media-path`` covering
the image directory, otherwise it refuses the ``file://`` URLs.

Usage:
    python scripts/regenerate_vlm_responses.py \
        --data ./output/visionarena/prompts.jsonl \
        --outfile ./output/visionarena/conversations.jsonl \
        --endpoint http://localhost:8000/v1/chat/completions
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
from tqdm import tqdm

# Reuse prepare_data.py's own adapter so the prompt this script sends is
# rendered from exactly the parts prepare_data.py will later render, instead of
# a second implementation that can drift from it.
from speculators.data_generation.preprocessing import _adapt_conv_for_vllm
from speculators.data_generation.vllm_client import (
    DEFAULT_MAX_RETRIES,
    InvalidResponseError,
    with_retries,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Transient statuses worth retrying: request timeout, conflict, too-early, and
# rate limiting, plus all 5xx. Other non-2xx replies (e.g. 400/401/404) are
# permanent config or client errors and fail fast.
SERVER_ERROR_STATUS = 500
RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate assistant responses for multimodal conversations using a "
            "vLLM-served target model, producing on-policy conversations JSONL."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Input JSONL of prompt-only conversations",
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        required=True,
        help="Output JSONL of conversations with on-policy assistant turns",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/v1/chat/completions",
        help=(
            "vLLM OpenAI-compatible Chat Completions endpoint. Note this is the "
            "full path, unlike prepare_data.py --render-endpoint which takes a "
            "base URL (default: http://localhost:8000/v1/chat/completions)"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name exposed by vLLM (auto-detected if not specified)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Stop after N input conversations"
    )
    parser.add_argument(
        "--concurrency", type=int, default=32, help="Max concurrent requests"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="max_tokens per generated turn (default: 2048)",
    )
    parser.add_argument(
        "--sampling-params",
        default=None,
        help=(
            "JSON object merged into each chat-completion request, e.g. "
            '\'{"temperature": 0.6, "top_p": 0.95}\'. Left empty, vLLM applies '
            "the model's own generation_config defaults, which is what serving "
            "would use."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to --outfile, skipping conversations it already contains",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            "Max retry attempts per request on transient failure "
            f"(default: {DEFAULT_MAX_RETRIES})"
        ),
    )
    args = parser.parse_args()

    if args.concurrency <= 0:
        parser.error("--concurrency must be > 0")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be > 0")
    try:
        args.sampling_params = (
            json.loads(args.sampling_params) if args.sampling_params else {}
        )
    except json.JSONDecodeError as e:
        parser.error(f"--sampling-params is not valid JSON: {e}")
    if not isinstance(args.sampling_params, dict):
        parser.error("--sampling-params must be a JSON object")
    return args


def prompt_turns(row: dict) -> list[dict]:
    """The system and user turns of an input row, in order.

    Assistant turns are dropped: whatever produced them is not the target model.
    """
    conversations = row.get("conversations") or row.get("messages") or []
    if not isinstance(conversations, list):
        return []

    turns: list[dict] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from")
        content = turn.get("content")
        if content is None:
            content = turn.get("value")
        if role in ("user", "human") and content:
            turns.append({"role": "user", "content": content})
        elif role == "system" and content:
            turns.append({"role": "system", "content": content})
    return turns


def missing_image_paths(turns: list[dict]) -> list[str]:
    """Local image paths referenced by ``turns`` that do not exist on disk.

    vLLM reports an unreadable ``file://`` URL as a generic bad request, so
    checking here turns a confusing 400 into a clear message naming the file.
    """
    missing: list[str] = []
    for turn in turns:
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            path = part.get("path")
            if path and not Path(path).is_file():
                missing.append(str(path))
    return missing


def assistant_turn(message: dict) -> dict:
    """Build the assistant turn to append from a chat-completion message.

    Reasoning content is preserved under both keys ``prepare_data.py``
    recognizes, so a thinking model's trace stays in the conversation rather
    than being dropped between regeneration and rendering.
    """
    turn: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    reasoning = message.get("reasoning_content")
    if reasoning:
        turn["thinking"] = reasoning
        turn["reasoning_content"] = reasoning
    return turn


def load_completed_ids(path: Path) -> set[str]:
    """Conversation ids already present in an output file, for ``--resume``."""
    completed: set[str] = set()
    if not path.is_file():
        return completed

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversation_id = row.get("conversation_id")
            if conversation_id:
                completed.add(str(conversation_id))
    return completed


def iter_input_rows(path: Path, limit: int | None) -> list[dict]:
    """Read the input JSONL, skipping malformed lines."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON on line %d", line_number)
                continue
            if limit is not None and len(rows) >= limit:
                break
    return rows


async def detect_model(endpoint: str) -> str:
    """Read the served model id off the vLLM server."""
    models_endpoint = endpoint.replace("/v1/chat/completions", "/v1/models")
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with (
            aiohttp.ClientSession(timeout=timeout) as session,
            session.get(models_endpoint) as response,
        ):
            data = await response.json()
            models = data.get("data", [])
            if not models:
                raise ValueError("No models found at endpoint")
            return models[0]["id"]
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(
            f"Failed to auto-detect model from {models_endpoint}: {e}. "
            f"Specify it with --model."
        ) from e


@with_retries
async def post_chat(
    session: aiohttp.ClientSession, endpoint: str, payload: dict
) -> dict:
    """POST one chat-completion request and return the parsed response.

    Transient failures raise ``RuntimeError`` so ``with_retries`` backs off and
    retries; permanent ones raise ``InvalidResponseError``, which it never
    retries.
    """
    async with session.post(endpoint, json=payload) as response:
        if not response.ok:
            body = (await response.text())[:500]
            message = f"HTTP {response.status} from {endpoint}: {body}"
            if (
                response.status >= SERVER_ERROR_STATUS
                or response.status in RETRYABLE_HTTP_STATUSES
            ):
                raise RuntimeError(message)
            raise InvalidResponseError(message)
        return await response.json()


async def regenerate_conversation(
    post_fn,
    turns: list[dict],
    *,
    model: str,
    max_tokens: int,
    sampling_params: dict,
) -> tuple[list[dict], bool]:
    """Regenerate every assistant turn of one conversation, in order.

    Two views of the growing conversation are kept: ``wire`` holds the
    ``image_url`` parts vLLM expects, and ``out`` keeps the ``path`` parts
    ``prepare_data.py`` expects. Returns ``(out, truncated)``; ``truncated`` is
    True when a response hit ``max_tokens``, in which case regeneration stops
    rather than conditioning later turns on a cut-off response.
    """
    out: list[dict] = []
    wire: list[dict] = []
    truncated = False

    for turn in turns:
        out.append(turn)
        wire.extend(_adapt_conv_for_vllm([turn]))
        if turn["role"] != "user":
            continue

        payload: dict[str, Any] = {
            # Spread first: the keys below are ours and must not be overridden.
            **sampling_params,
            "model": model,
            "messages": wire,
            "max_tokens": max_tokens,
        }
        data = await post_fn(payload)

        choice = data["choices"][0]
        message = choice["message"]
        generated = assistant_turn(message)
        if not generated["content"] and not generated.get("reasoning_content"):
            raise ValueError("empty assistant generation")

        out.append(generated)
        wire.append(generated)

        if choice.get("finish_reason") == "length":
            truncated = True
            break

    return out, truncated


async def worker(
    session: aiohttp.ClientSession,
    queue: "asyncio.Queue",
    args: argparse.Namespace,
    out_handle,
    err_handle,
    progress,
    stats: dict,
) -> None:
    """Pull conversations off the queue and write regenerated rows."""

    async def post(payload: dict) -> dict:
        started = time.perf_counter()
        result = await post_chat(
            session, args.endpoint, payload, max_retries=args.max_retries
        )
        stats["total_request_s"] += time.perf_counter() - started
        stats["requests"] += 1
        return result

    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return

        try:
            conversations, truncated = await regenerate_conversation(
                post,
                item["turns"],
                model=args.model,
                max_tokens=args.max_tokens,
                sampling_params=args.sampling_params,
            )
            row = {
                "conversation_id": item["conversation_id"],
                "conversations": conversations,
                "metadata": {"model": args.model, "truncated": truncated},
            }
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_handle.flush()
            stats["ok"] += 1
            stats["truncated"] += truncated
        except Exception as e:  # noqa: BLE001
            # Failures go to a separate file so the training input stays clean.
            err_handle.write(
                json.dumps(
                    {"conversation_id": item["conversation_id"], "error": repr(e)},
                    ensure_ascii=False,
                )
                + "\n"
            )
            err_handle.flush()
            stats["errors"] += 1
        finally:
            progress.set_postfix(
                {
                    "ok": stats["ok"],
                    "err": stats["errors"],
                    "trunc": stats["truncated"],
                },
                refresh=False,
            )
            progress.update(1)
            queue.task_done()


def log_summary(stats: dict) -> None:
    elapsed = time.perf_counter() - stats["start_time"]
    logger.info(
        "Regenerated %d conversations in %.1fs (%d errors, %d truncated)",
        stats["ok"],
        elapsed,
        stats["errors"],
        stats["truncated"],
    )
    if stats["requests"]:
        logger.info(
            "%d requests, %.1f req/s, avg latency %.0f ms",
            stats["requests"],
            stats["requests"] / elapsed if elapsed else 0.0,
            stats["total_request_s"] / stats["requests"] * 1000,
        )


def build_work_items(rows: list[dict], completed: set[str]) -> list[dict]:
    """Turn input rows into regeneration work items, dropping unusable ones.

    Rows already present in the output file, rows with no user turn, and rows
    whose images are not on disk are skipped.
    """
    items: list[dict] = []
    num_missing_images = 0

    for index, row in enumerate(rows):
        conversation_id = str(row.get("conversation_id") or f"row{index}")
        if conversation_id in completed:
            continue
        turns = prompt_turns(row)
        if not turns:
            continue
        if missing := missing_image_paths(turns):
            if num_missing_images == 0:
                logger.error(
                    "Skipping conversations whose images are missing from disk, "
                    "first is %s. Re-run the export step to materialize them.",
                    missing[0],
                )
            num_missing_images += 1
            continue
        items.append({"conversation_id": conversation_id, "turns": turns})

    if num_missing_images:
        logger.warning(
            "Skipped %d conversations with missing images", num_missing_images
        )
    return items


async def main() -> None:
    args = parse_args()

    if args.model is None:
        args.model = await detect_model(args.endpoint)
    logger.info("Regenerating with model %s via %s", args.model, args.endpoint)

    rows = iter_input_rows(args.data, args.limit)
    logger.info("Read %d conversations from %s", len(rows), args.data)

    completed = load_completed_ids(args.outfile) if args.resume else set()
    if completed:
        logger.info("Resuming: %d conversations already done", len(completed))

    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    error_outfile = args.outfile.with_suffix(f".errors{args.outfile.suffix}")

    items = build_work_items(rows, completed)
    if not items:
        logger.warning("Nothing to regenerate")
        return

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    stats = {
        "ok": 0,
        "errors": 0,
        "truncated": 0,
        "requests": 0,
        "total_request_s": 0.0,
        "start_time": time.perf_counter(),
    }

    # No total read timeout: a long multimodal prefill on a busy server is slow,
    # not stuck, and cancelling it mid-flight only wastes the work.
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=90, sock_read=None)
    # One connection per worker: concurrency is already bounded by the pool of
    # workers, so a larger limit would only leave sockets idle.
    connector = aiohttp.TCPConnector(limit=args.concurrency, enable_cleanup_closed=True)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        with (
            args.outfile.open("a" if args.resume else "w", encoding="utf-8") as out_h,
            error_outfile.open("a" if args.resume else "w", encoding="utf-8") as err_h,
            tqdm(total=len(items), desc="Regenerating", unit="conv") as progress,
        ):
            workers = [
                asyncio.create_task(
                    worker(session, queue, args, out_h, err_h, progress, stats)
                )
                for _ in range(args.concurrency)
            ]
            for item in items:
                await queue.put(item)
            for _ in workers:
                await queue.put(None)
            await asyncio.gather(*workers)

    log_summary(stats)
    logger.info("On-policy conversations written to %s", args.outfile)
    logger.info("Next: pass it to prepare_data.py --data %s", args.outfile)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
