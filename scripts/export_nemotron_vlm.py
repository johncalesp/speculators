#!/usr/bin/env python3
"""
Export Llama-Nemotron-VLM-Dataset-v1 prompts and images for on-policy VLM data.

``nvidia/Llama-Nemotron-VLM-Dataset-v1`` is a 2.86M-row VLM SFT dataset split
into 21 partitions (``ocr_1``..``ocr_10``, ``vqa_1``..``vqa_9``,
``captioning_1``..``captioning_2``). Its assistant turns come from other models,
so -- exactly as with VisionArena -- this script keeps the user prompts and
their images, drops the responses, and leaves regeneration to
``regenerate_vlm_responses.py``.

The layout differs from VisionArena in three ways that shape this script:

1. Each partition is a JSONL file, not a parquet shard, with rows of the form
   ``{"id", "image", "conversations": [{"from": "human"|"gpt", "value"}]}``.
   The user turn carries an ``<image>`` placeholder that marks where the image
   belongs in the text.
2. Images live in webdataset TAR shards under ``<partition>_images/``, keyed by
   member name. The ``image`` field is that member name verbatim, so the tars
   can be read directly and megatron-energon is not needed.
3. **Only the partitions whose images were downloaded are usable.** The repo
   ships every partition's JSONL but only some partitions' images; the rest are
   commented out in ``metadataset.yaml``. ``--list-partitions`` reports which
   are which, and a partition without images is rejected rather than exported
   as prompts pointing at absent files.

Sampling is by fraction rather than row count: ``--fraction 0.1`` keeps a tenth
of each requested partition. Selection is a hash threshold on the row id, not a
shuffle, which makes it

* deterministic -- the same rows for the same ``--seed``, no state to carry;
* monotonic -- raising ``--fraction`` keeps every previously exported row and
  adds to it, so scaling up reuses the images already extracted instead of
  resampling and re-extracting a different tenth.

The output matches ``export_visionarena.py``'s, so the rest of the pipeline is
unchanged:

    {"conversation_id": "22935085...", "conversations": [
        {"role": "user", "content": [
            {"type": "image", "path": "/abs/path/ocr_1/502450.png"},
            {"type": "text", "text": "Extract all visible text."}]}]}

Usage:
    # what is actually usable in the local copy
    python scripts/export_nemotron_vlm.py --list-partitions

    # 10% of two partitions
    python scripts/export_nemotron_vlm.py \
        --partitions ocr_1,vqa_9 --fraction 0.1 \
        --image-dir ./output/nemotron/images \
        --outfile ./output/nemotron/prompts.jsonl
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

for _noisy in ("httpx", "httpcore", "urllib3", "hf_xet", "filelock", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

NEMOTRON_HF_PATH = "nvidia/Llama-Nemotron-VLM-Dataset-v1"

# The placeholder the dataset uses to mark where the image sits in the prompt.
IMAGE_PLACEHOLDER = "<image>"

# Roles use the ShareGPT/LLaVA spelling rather than OpenAI's.
_ROLES = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant"}

# A filename has to stay under the filesystem's limit once flattened; longer
# names fall back to a digest.
_MAX_FILENAME = 200

# Partitions whose absent images `download_nemotron_images.py` can fetch, named
# here only to point at it in the error for an imageless partition. Kept in step
# with that script's IMAGE_SOURCES by a test.
_DOWNLOADABLE = {"captioning_1", "captioning_2", "vqa_1", "vqa_2", "vqa_3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Llama-Nemotron-VLM-Dataset-v1 user prompts and images as "
            "prompt-only conversations for on-policy response regeneration."
        )
    )
    parser.add_argument(
        "--outfile",
        type=Path,
        default=Path("./output/nemotron_vlm/prompts.jsonl"),
        help=(
            "JSONL path for the exported prompt-only conversations "
            "(default: ./output/nemotron_vlm/prompts.jsonl)"
        ),
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=(
            "Directory to extract images into, one subdirectory per partition "
            "(default: 'images' next to --outfile). Serve vLLM with "
            "--allowed-local-media-path pointing here."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Directory holding the already-downloaded dataset. Default: the "
            "cached snapshot in the HuggingFace cache ($HF_HOME, else "
            "~/.cache/huggingface). Nothing is downloaded."
        ),
    )
    parser.add_argument(
        "--image-source",
        type=Path,
        default=None,
        help=(
            "Directory holding the <partition>_images/ directories. Defaults to "
            "--dataset-path. Set it to the --image-source given to "
            "download_nemotron_images.py when images were fetched outside the "
            "dataset directory."
        ),
    )
    parser.add_argument(
        "--partitions",
        type=str,
        default=None,
        help=(
            "Comma-separated partitions to export, e.g. 'ocr_1,ocr_4,vqa_9'. "
            "Default: every partition whose images are present. Partitions "
            "without downloaded images are rejected -- see --list-partitions."
        ),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of each partition's rows to keep: 1 = all, 0.5 = half, "
            "0.01 = one percent (default: 1). Selection is a hash threshold, so "
            "raising it later keeps the rows already exported and only adds to "
            "them."
        ),
    )
    parser.add_argument(
        "--list-partitions",
        action="store_true",
        help=(
            "Print each partition, its row count and whether its images are "
            "present, then exit. Reads no rows."
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
        help=(
            "Salts the selection hash. Changing it selects a different subset of "
            "the same size, so leave it alone when topping up (default: 0)."
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.image_dir is None:
        args.image_dir = args.outfile.parent / "images"
    return args


def resolve_dataset_dir(dataset_path: Path | None) -> Path:
    """Locate the already-downloaded dataset without downloading it."""
    if dataset_path is not None:
        if not dataset_path.is_dir():
            raise FileNotFoundError(
                f"--dataset-path {dataset_path} is not a directory."
            )
        return dataset_path

    try:
        snapshot = snapshot_download(
            NEMOTRON_HF_PATH, repo_type="dataset", local_files_only=True
        )
    except Exception as exc:  # hub raises several types here
        raise FileNotFoundError(
            f"{NEMOTRON_HF_PATH} is not in the HuggingFace cache "
            f"({os.getenv('HF_HOME') or '~/.cache/huggingface'}). Download it "
            f"with:\n    hf download {NEMOTRON_HF_PATH} --repo-type dataset\n"
            "or point --dataset-path at a directory that already holds it."
        ) from exc
    return Path(snapshot)


def count_rows(partition_file: Path) -> int:
    """Row count for a partition, from its sidecar index when there is one.

    The dataset ships a ``.jsonl.idx`` next to each partition: little-endian
    uint64 byte offsets, one per row plus a terminator. Reading its size beats
    counting newlines through gigabytes of JSON.
    """
    index = partition_file.with_suffix(".jsonl.idx")
    if index.is_file():
        return max(0, os.path.getsize(os.path.realpath(index)) // 8 - 1)

    with partition_file.open("rb") as handle:
        return sum(1 for _ in handle)


def partition_image_dir(image_source: Path, partition: str) -> Path:
    """Where a partition's images live under ``image_source``.

    Shared with ``download_nemotron_images.py`` so a download and the export
    that consumes it agree on the layout without either passing paths around.
    """
    return image_source / f"{partition}_images"


def image_roots(dataset_dir: Path, image_source: Path | None) -> list[Path]:
    """Directories to look for ``<partition>_images/`` in, most specific first.

    ``--image-source`` adds a location rather than replacing the dataset
    directory: the partitions the repo ships images for keep their tars beside
    the JSONL, so treating the flag as an override would hide them the moment
    it was used to point at separately downloaded images.
    """
    roots = [dataset_dir] if image_source is None else [image_source, dataset_dir]
    seen: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.append(root)
    return seen


def image_shards(dataset_dir: Path, partition: str) -> list[Path]:
    """The TAR shards holding a partition's images, empty if none were downloaded."""
    image_dir = partition_image_dir(dataset_dir, partition)
    if not image_dir.is_dir():
        return []
    return sorted(image_dir.glob("*.tar"))


def loose_image_dir(root: Path, partition: str) -> Path | None:
    """The partition's image directory if it holds plain image files, else None.

    The partitions the repo ships images for use TAR shards, but the ones whose
    images come from an external source (OpenImages and friends, fetched by
    ``download_nemotron_images.py``) are plain files named by the ``image``
    field. Any non-tar file counts, and the scan stops at the first one: these
    directories hold up to a million entries.
    """
    image_dir = partition_image_dir(root, partition)
    if not image_dir.is_dir():
        return None
    for entry in image_dir.iterdir():
        if entry.suffix not in {".tar", ".partial"} and entry.is_file():
            return image_dir
    return None


def discover_partitions(
    dataset_dir: Path, image_source: Path | None = None
) -> list[dict[str, Any]]:
    """Every partition in the local copy, with row count and image availability."""
    roots = image_roots(dataset_dir, image_source)
    partitions = []
    for partition_file in sorted(dataset_dir.glob("*.jsonl")):
        partition = partition_file.stem

        shards: list[Path] = []
        loose: Path | None = None
        for root in roots:
            shards = image_shards(root, partition)
            if shards:
                break
            loose = loose_image_dir(root, partition)
            if loose is not None:
                break

        partitions.append(
            {
                "name": partition,
                "path": partition_file,
                "rows": count_rows(partition_file),
                "shards": shards,
                "loose_dir": loose,
                "has_images": bool(shards) or loose is not None,
            }
        )
    return partitions


def select_partitions(
    dataset_dir: Path, requested: str | None, image_source: Path | None = None
) -> list[dict[str, Any]]:
    """Resolve --partitions against the local copy, rejecting unusable ones.

    A partition whose images were not downloaded is an error rather than a
    warning: exporting it would produce prompts referencing files that do not
    exist, and the failure would only surface later as vLLM refusing to fetch
    them, one request at a time.
    """
    available = {
        part["name"]: part for part in discover_partitions(dataset_dir, image_source)
    }
    if not available:
        raise FileNotFoundError(
            f"No *.jsonl partitions found under {dataset_dir}. Point "
            "--dataset-path at the dataset directory, or download it with\n"
            f"    hf download {NEMOTRON_HF_PATH} --repo-type dataset"
        )

    if requested is None:
        chosen = [part for part in available.values() if part["has_images"]]
        if not chosen:
            raise FileNotFoundError(
                f"None of the {len(available)} partitions under {dataset_dir} "
                "have their images downloaded, so there is nothing to export. "
                "Run with --list-partitions to see the inventory."
            )
        logger.info(
            "No --partitions given; using all %d with images: %s",
            len(chosen),
            ", ".join(part["name"] for part in chosen),
        )
        return chosen

    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--partitions was empty")

    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown partition(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}"
        )

    imageless = [name for name in names if not available[name]["has_images"]]
    if imageless:
        with_images = sorted(n for n, p in available.items() if p["has_images"])
        downloadable = sorted(set(imageless) & _DOWNLOADABLE)
        hint = ""
        if downloadable:
            hint = (
                f"\nImages for {', '.join(downloadable)} are on OpenImages, "
                "addressable from each row's image field, so they can be "
                "fetched:\n"
                f"    python scripts/download_nemotron_images.py "
                f"--partitions {','.join(downloadable)} --fraction <f> --dry-run"
            )
        raise ValueError(
            f"Partition(s) {', '.join(imageless)} have no images in "
            f"{image_source or dataset_dir}: the dataset ships every partition's "
            "JSONL but only some partitions' image shards, and the rest are "
            "commented out in metadataset.yaml. Exporting them would yield "
            "prompts pointing at files that do not exist.\n"
            f"Partitions with images here: {', '.join(with_images) or '(none)'}"
            f"{hint}"
        )

    return [available[name] for name in names]


def keeps_row(partition: str, row_id: str, fraction: float, seed: int) -> bool:
    """Whether a row survives fraction sampling.

    A digest of the row id is read as a fraction of the 64-bit range and
    compared against ``fraction``. Uniform because the ids are UUIDs, and
    nested in ``fraction``: every row kept at 0.1 is also kept at 0.2, so
    raising the fraction adds rows instead of replacing them. The partition
    name is salted in so partitions do not share a selection pattern.
    """
    if fraction >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{partition}:{row_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < fraction * (1 << 64)


def destination_name(member_name: str) -> str:
    """A flat, traversal-safe filename for a TAR member.

    Member names can be nested (``data/train/project-26/0000160/99833.md.jpg``),
    so separators are folded rather than recreated as directories: a name from
    the archive must not be able to decide where the file lands. Overlong
    results fall back to a digest, keeping the suffix so the format stays
    detectable.
    """
    flat = member_name.strip("/").replace("/", "__")
    # '.' and '..' survive the fold unchanged and would resolve to the image
    # directory itself or its parent. Path('..').name is '..', not '', so a
    # basename comparison alone does not catch them -- they are matched by name.
    if flat in {"", ".", ".."} or flat != Path(flat).name:
        flat = hashlib.sha256(member_name.encode()).hexdigest()[:32]
    if len(flat) > _MAX_FILENAME:
        suffix = Path(flat).suffix[:10]
        flat = hashlib.sha256(member_name.encode()).hexdigest()[:32] + suffix
    return flat


def prompt_parts(value: str, image_path: Path) -> list[dict]:
    """Split a user turn on ``<image>`` into ordered content parts.

    The placeholder marks where the image belongs, so its position is preserved
    rather than assuming it leads: a prompt reading "compare <image> with the
    table" would otherwise be reordered.
    """
    parts: list[dict] = []
    for index, chunk in enumerate(value.split(IMAGE_PLACEHOLDER)):
        if index > 0:
            parts.append({"type": "image", "path": str(image_path)})
        text = chunk.strip()
        if text:
            parts.append({"type": "text", "text": text})
    return parts


def extract_prompt_turns(conversations: Any) -> list[dict]:
    """Keep the system and user turns, dropping the dataset's responses."""
    if not isinstance(conversations, list):
        return []

    turns: list[dict] = []
    for message in conversations:
        if not isinstance(message, dict):
            continue
        sender = message.get("from")
        role = _ROLES.get(sender) if isinstance(sender, str) else None
        value = message.get("value")
        if role == "user":
            if not value:
                # Nothing to answer, and keeping it would misalign the image.
                return []
            turns.append({"role": "user", "value": value})
        elif role == "system" and value:
            turns.append({"role": "system", "value": value})

    if not any(turn["role"] == "user" for turn in turns):
        return []
    while turns and turns[-1]["role"] == "system":
        turns.pop()
    return turns


def build_row(row: dict, partition: str, image_path: Path) -> dict | None:
    """Convert one dataset row into a prompt-only conversation, or skip it."""
    turns = extract_prompt_turns(row.get("conversations"))
    if not turns:
        return None

    conversations: list[dict] = []
    attached = False
    for turn in turns:
        if turn["role"] != "user":
            conversations.append({"role": "system", "content": turn["value"]})
            continue
        if attached or IMAGE_PLACEHOLDER not in turn["value"]:
            text = turn["value"].replace(IMAGE_PLACEHOLDER, "").strip()
            if not text:
                return None
            conversations.append({"role": "user", "content": text})
            continue
        parts = prompt_parts(turn["value"], image_path)
        if not any(part["type"] == "text" for part in parts):
            # An image with no question gives the model nothing to answer.
            return None
        conversations.append({"role": "user", "content": parts})
        attached = True

    if not attached:
        return None
    return {
        "conversation_id": f"{partition}/{row.get('id')}",
        "conversations": conversations,
    }


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


def select_rows(
    partition: dict[str, Any],
    fraction: float,
    seed: int,
    exported_ids: set[str],
) -> dict[str, list[dict]]:
    """Rows of a partition that sampling keeps, grouped by image member name.

    Grouping by image lets the TAR pass emit every row for an image the moment
    it is extracted, and costs nothing when images are unique -- which they are
    in this dataset.
    """
    name = partition["name"]
    wanted: dict[str, list[dict]] = defaultdict(list)
    with partition["path"].open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            member = row.get("image")
            if not row_id or not isinstance(member, str) or not member:
                continue
            if not keeps_row(name, str(row_id), fraction, seed):
                continue
            if f"{name}/{row_id}" in exported_ids:
                continue
            wanted[member].append(row)
    return wanted


def export_loose_partition(
    partition: dict[str, Any],
    wanted: dict[str, list[dict]],
    handle,
    progress: tqdm,
) -> tuple[int, int, int]:
    """Write conversations for a partition whose images are already plain files.

    The images are referenced where they sit instead of being copied into the
    export's image directory. Copying would be a second full copy of a selection
    that reaches 378 GB for vqa_1, for no benefit: nothing downstream writes to
    these files, and the downloader already placed them atomically.
    """
    name = partition["name"]
    image_dir = partition["loose_dir"]

    num_written = 0
    num_skipped = 0
    num_missing = 0

    for member, rows in wanted.items():
        # The member name comes from the dataset, not from us, so it is confined
        # to the image directory before use.
        candidate = (image_dir / member).resolve()
        if image_dir.resolve() not in candidate.parents or not candidate.is_file():
            num_missing += len(rows)
            continue
        written, skipped = _write_rows(rows, name, candidate, handle, progress)
        num_written += written
        num_skipped += skipped

    return num_written, num_skipped, num_missing


def export_partition(
    partition: dict[str, Any],
    wanted: dict[str, list[dict]],
    image_root: Path,
    handle,
    progress: tqdm,
) -> tuple[int, int, int]:
    """Extract the wanted images and write their conversations.

    Shards are read in one sequential pass each, which is why selection happens
    first: seeking to individual members across tens of GB of TAR would cost far
    more than streaming past the members we do not want.
    """
    if partition.get("loose_dir") is not None:
        return export_loose_partition(partition, wanted, handle, progress)

    name = partition["name"]
    image_dir = image_root / name
    image_dir.mkdir(parents=True, exist_ok=True)

    num_written = 0
    num_skipped = 0
    remaining = set(wanted)

    for shard in partition["shards"]:
        if not remaining:
            break
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                if not remaining:
                    break
                if not member.isfile() or member.name not in remaining:
                    continue
                remaining.discard(member.name)

                destination = image_dir / destination_name(member.name)
                if not destination.exists() and not _extract_member(
                    archive, member, destination
                ):
                    continue

                written, skipped = _write_rows(
                    wanted[member.name], name, destination, handle, progress
                )
                num_written += written
                num_skipped += skipped

    num_missing = sum(len(wanted[member]) for member in remaining)
    return num_written, num_skipped, num_missing


def _extract_member(archive: tarfile.TarFile, member, destination: Path) -> bool:
    """Write one TAR member to ``destination``, atomically."""
    extracted = archive.extractfile(member)
    if extracted is None:
        return False
    # Written via a temp name so an interrupted run cannot leave a truncated
    # image that later looks already-extracted.
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(extracted.read())
    partial.replace(destination)
    return True


def _write_rows(
    rows: list[dict], partition: str, image: Path, handle, progress: tqdm
) -> tuple[int, int]:
    """Write every usable row for one image. Returns (written, skipped)."""
    num_written = 0
    num_skipped = 0
    for row in rows:
        exported = build_row(row, partition, image)
        if exported is None:
            num_skipped += 1
            continue
        handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
        num_written += 1
        progress.update(1)
    return num_written, num_skipped


def media_roots(partitions: list[dict[str, Any]], image_dir: Path) -> set[Path]:
    """The directories vLLM has to be allowed to read images from.

    Images from tars are extracted under ``image_dir``, but downloaded ones are
    referenced where they already sit, so the roots are derived rather than
    assumed: pointing vLLM at ``image_dir`` alone would make it refuse every
    downloaded image. A root inside another is dropped as already covered,
    which is the normal case when the images were downloaded under
    ``image_dir``.
    """
    candidates = {
        part["loose_dir"].resolve()
        for part in partitions
        if part["loose_dir"] is not None
    }
    if any(part["loose_dir"] is None for part in partitions):
        candidates.add(image_dir.resolve())
    return {
        root
        for root in candidates
        if not any(other in root.parents for other in candidates)
    }


def report_media_roots(partitions: list[dict[str, Any]], image_dir: Path) -> None:
    """Log where images will be and what vLLM must be allowed to read."""
    if any(part["loose_dir"] is None for part in partitions):
        logger.info("Extracting images to %s", image_dir.resolve())

    roots = media_roots(partitions, image_dir)
    for root in sorted(roots):
        logger.info("Images referenced under %s", root)

    if len(roots) == 1:
        logger.info(
            "Serve vLLM with --allowed-local-media-path %s so it can read them",
            next(iter(roots)),
        )
    else:
        logger.warning(
            "Images live under %d roots (%s). vLLM accepts one "
            "--allowed-local-media-path, so point it at a common parent, or "
            "rerun the download with --image-source %s to keep everything in "
            "one place.",
            len(roots),
            ", ".join(str(root) for root in sorted(roots)),
            image_dir.resolve(),
        )


def print_partition_table(dataset_dir: Path, image_source: Path | None = None) -> None:
    partitions = discover_partitions(dataset_dir, image_source)
    if not partitions:
        raise FileNotFoundError(f"No *.jsonl partitions found under {dataset_dir}")

    total_rows = sum(part["rows"] for part in partitions)
    usable_rows = sum(part["rows"] for part in partitions if part["has_images"])
    print(f"\n{NEMOTRON_HF_PATH}\n{dataset_dir}\n")
    print(f"{'partition':16s} {'rows':>10s}  {'images':>7s}  {'source':>9s}  size")
    print("-" * 62)
    for part in partitions:
        size = sum(os.path.getsize(os.path.realpath(s)) for s in part["shards"])
        if part["shards"]:
            source = f"{len(part['shards'])} tars"
            size_text = f"{size / 1e9:6.1f} GB"
        elif part["loose_dir"] is not None:
            source = "files"
            size_text = "downloaded"
        else:
            source = "-"
            size_text = ""
        print(
            f"{part['name']:16s} {part['rows']:10,d}  "
            f"{'YES' if part['has_images'] else 'no':>7s}  "
            f"{source:>9s}  {size_text}"
        )
    print("-" * 62)
    print(f"{'total':16s} {total_rows:10,d}")
    print(f"{'exportable':16s} {usable_rows:10,d}  (partitions with images)\n")
    print(
        "Partitions marked 'no' ship their JSONL but not their images; they are\n"
        "the entries commented out in metadataset.yaml. Five of them draw on\n"
        "OpenImages and can be fetched with scripts/download_nemotron_images.py\n"
        f"({', '.join(sorted(_DOWNLOADABLE))}); the rest need their source\n"
        "dataset obtained by hand -- see <partition>.md.\n"
    )


def main() -> None:
    args = parse_args()

    dataset_dir = resolve_dataset_dir(args.dataset_path)
    image_source = args.image_source or dataset_dir

    if args.list_partitions:
        print_partition_table(dataset_dir, image_source)
        return

    partitions = select_partitions(dataset_dir, args.partitions, image_source)

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)

    exported_ids = load_exported_ids(args.outfile) if args.resume else set()
    if exported_ids:
        logger.info(
            "Resuming: %d conversations already in %s", len(exported_ids), args.outfile
        )

    logger.info("Reading local copy of %s from %s", NEMOTRON_HF_PATH, dataset_dir)
    logger.info(
        "Partitions: %s", ", ".join(f"{p['name']}({p['rows']:,})" for p in partitions)
    )
    report_media_roots(partitions, args.image_dir)

    # Selection is resolved for every partition up front so the run can report
    # what it is about to do, and so a bad partition fails before any work.
    logger.info("Selecting rows at fraction=%.4g (seed=%d)", args.fraction, args.seed)
    selections = []
    total_selected = 0
    for partition in partitions:
        wanted = select_rows(partition, args.fraction, args.seed, exported_ids)
        num_rows = sum(len(rows) for rows in wanted.values())
        selections.append((partition, wanted))
        total_selected += num_rows
        logger.info(
            "  %-14s %8d of %8d rows (%5.2f%%)",
            partition["name"],
            num_rows,
            partition["rows"],
            100 * num_rows / partition["rows"] if partition["rows"] else 0.0,
        )

    if total_selected == 0:
        logger.info("Nothing new to export; output already holds every selected row.")
        return
    logger.info("Exporting %d conversations", total_selected)

    num_written = num_skipped = num_missing = 0
    with (
        args.outfile.open("a" if args.resume else "w", encoding="utf-8") as handle,
        tqdm(total=total_selected, desc="Exporting", unit="conv") as progress,
    ):
        for partition, wanted in selections:
            written, skipped, missing = export_partition(
                partition, wanted, args.image_dir, handle, progress
            )
            num_written += written
            num_skipped += skipped
            num_missing += missing
        handle.flush()

    logger.info(
        "Exported %d conversations to %s (%d now in file)",
        num_written,
        args.outfile,
        len(exported_ids) + num_written,
    )
    if num_skipped:
        logger.info("Skipped %d rows as unusable", num_skipped)
    if num_missing:
        logger.warning(
            "%d selected rows were dropped because their image was not found. "
            "For a partition read from TAR shards this means an incomplete "
            "download; for downloaded images it is usually a handful of keys "
            "the upstream host has removed, which download_nemotron_images.py "
            "reports as 404s.",
            num_missing,
        )
    logger.info(
        "Next: generate on-policy responses with scripts/regenerate_vlm_responses.py"
    )


if __name__ == "__main__":
    main()
    # Same teardown guard as export_visionarena.py: flush everything and leave
    # before interpreter finalization, so a native reader thread cannot turn a
    # finished export into a non-zero exit and stop the pipeline.
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
