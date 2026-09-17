#!/usr/bin/env python3
"""Convert JSON/JSONL chat request bodies into prompts for DFlash training.

This reads data, never shell commands. Base64 images and local image files are
validated and copied into a content-addressed image directory. Input is streamed
one record at a time for JSONL; .json accepts a single request or a request list.
"""

import argparse
import base64
import binascii
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

# Transport settings, credentials and arbitrary API extensions are not forwarded.
GENERATION_FIELDS = {
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "max_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
}
IGNORED_FIELDS = {"id", "conversation_id", "model", "messages", "stream"}


def input_records(path: Path) -> Iterator[tuple[str, dict]]:
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        for index, row in enumerate(value if isinstance(value, list) else [value], 1):
            yield f"{path}:{index}", row
    else:
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        yield f"{path}:{number}", json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{number}: invalid JSON") from error


def materialize_image(url: str, source_dir: Path, image_dir: Path) -> str:
    if not isinstance(url, str):
        raise ValueError("image_url.url must be a string")
    if url.startswith("data:"):
        header, separator, encoded = url.partition(",")
        if (
            not separator
            or not header.startswith("data:image/")
            or not header.endswith(";base64")
        ):
            raise ValueError("Expected data:image/<format>;base64,<encoded bytes>")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Invalid base64 image") from error
    else:
        parsed = urlparse(url)
        if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
            raise ValueError(
                "Use base64 images or local files; download HTTP images first"
            )
        path = Path(unquote(parsed.path)) if parsed.scheme else Path(url)
        data = (source_dir / path).read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as img:
            suffix = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}.get(img.format)
            if suffix is None:
                raise ValueError("Only PNG, JPEG and WEBP images are supported")
            img.verify()
    except (OSError, SyntaxError) as error:
        raise ValueError("Invalid image bytes") from error
    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / (hashlib.sha256(data).hexdigest() + suffix)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return str(destination.resolve())


def normalize_content(content, source_dir: Path, image_dir: Path):
    if isinstance(content, str) and content.strip():
        return content
    if not isinstance(content, list) or not content:
        raise ValueError("Message content must be nonempty text or a list of parts")
    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Content parts must be objects")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            parts.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            if not isinstance(image, dict) or set(image) - {"url", "detail"}:
                raise ValueError("image_url must contain url and optional detail")
            if image.get("detail", "auto") != "auto":
                raise ValueError(
                    "Only image detail=auto is supported; set MM_PROCESSOR_KWARGS"
                )
            parts.append(
                {
                    "type": "image",
                    "path": materialize_image(image.get("url"), source_dir, image_dir),
                }
            )
        else:
            raise ValueError(f"Unsupported content part type: {part.get('type')!r}")
    return parts


def convert_request(request: dict, source_dir: Path, image_dir: Path) -> dict:
    if not isinstance(request, dict):
        raise ValueError("Each record must be a chat request object")
    if unknown := set(request) - GENERATION_FIELDS - IGNORED_FIELDS:
        raise ValueError(f"Unsupported request fields: {sorted(unknown)}")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Each request needs a nonempty messages list")
    turns = []
    previous = None
    for message in messages:
        if not isinstance(message, dict) or set(message) - {"role", "content"}:
            raise ValueError("Messages must contain only role and content")
        role = message.get("role")
        if role == "assistant" and previous == "user":
            # Regenerate assistant history with the target, as for VisionArena.
            previous = role
            continue
        if (
            role == "system"
            and not turns
            or role == "user"
            and previous in (None, "system", "assistant")
        ):
            turns.append(
                {
                    "role": role,
                    "content": normalize_content(
                        message.get("content"), source_dir, image_dir
                    ),
                }
            )
        else:
            raise ValueError(
                "Expected optional system, then alternating user/assistant turns"
            )
        previous = role
    if not any(turn["role"] == "user" for turn in turns):
        raise ValueError("Request needs at least one user message")
    params = {key: request[key] for key in GENERATION_FIELDS if key in request}
    if "max_tokens" in params and (
        type(params["max_tokens"]) is not int or params["max_tokens"] <= 0
    ):
        raise ValueError("max_tokens must be a positive integer")
    # Hash normalized content as well as the caller's ID: an edited request must
    # never reuse an old response just because its external ID stayed the same.
    identity = {
        "turns": turns,
        "generation_params": params,
        "id": request.get("id", request.get("conversation_id")),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {
        "conversation_id": f"requests/{digest}",
        "conversations": turns,
        "generation_params": params,
    }


def export_requests(
    inputs: list[Path], outfile: Path, image_dir: Path, limit: int | None = None
) -> int:
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_suffix(outfile.suffix + ".tmp")
    seen = set()
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for path in inputs:
                for location, request in input_records(path):
                    try:
                        row = convert_request(request, path.resolve().parent, image_dir)
                    except (ValueError, OSError) as error:
                        raise ValueError(f"{location}: {error}") from error
                    if row["conversation_id"] not in seen:
                        seen.add(row["conversation_id"])
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if limit is not None and len(seen) >= limit:
                        break
                if limit is not None and len(seen) >= limit:
                    break
        if not seen:
            raise ValueError("No requests found")
        temporary.replace(outfile)
    finally:
        temporary.unlink(missing_ok=True)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        count = export_requests(args.input, args.outfile, args.image_dir, args.limit)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Exported {count} unique requests to {args.outfile}")


if __name__ == "__main__":
    main()
