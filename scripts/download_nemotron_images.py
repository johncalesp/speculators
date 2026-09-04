#!/usr/bin/env python3
"""
Download the externally-hosted images for Llama-Nemotron-VLM-Dataset-v1.

``nvidia/Llama-Nemotron-VLM-Dataset-v1`` ships every partition's JSONL but only
some partitions' image tars. The rest name images that live in the datasets they
were annotated from, and five of those partitions draw on OpenImages, whose URLs
are derivable from the ``image`` field alone:

    captioning_1  captioning_2  vqa_1  vqa_2  vqa_3

This script fetches those images so ``export_nemotron_vlm.py`` can use the
partitions. The others (ChartQA, DocLayNet, PubTables-1M, ...) need their source
datasets obtained by hand and are rejected with a pointer to the partition's
``.md``.

**Sampling is shared with the export.** ``vqa_1`` alone is 1,278,221 images and
about 378 GB, and a run that only trains on a tenth of it has no use for the
other 340 GB. Passing the same ``--fraction`` and ``--seed`` here as to
``export_nemotron_vlm.py`` downloads exactly the images that export will ask
for -- both call the same ``keeps_row``, so the two selections cannot drift.
Because that sampling is monotonic, raising the fraction later downloads only
the newly selected images and keeps the ones already on disk.

Downloads are resumable: a file already present is never refetched, so an
interrupted run is continued by rerunning the same command.

Images land in ``<image-source>/<partition>_images/`` -- the layout the export
looks in, so a download followed by an export needs no path wiring. They are
referenced from there in place rather than copied, which is why this directory
should sit on a disk with room for the full selection.

Usage:
    # size the download before committing to it
    python scripts/download_nemotron_images.py \
        --partitions vqa_1 --fraction 0.05 --dry-run

    # fetch it
    python scripts/download_nemotron_images.py --partitions vqa_1 --fraction 0.05

    # then export the same selection
    python scripts/export_nemotron_vlm.py --partitions vqa_1 --fraction 0.05
"""

import argparse
import asyncio
import json
import logging
import os
import random
import sys
from pathlib import Path

import aiohttp
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_nemotron_vlm import (
    NEMOTRON_HF_PATH,
    count_rows,
    keeps_row,
    partition_image_dir,
    resolve_dataset_dir,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

for _noisy in (
    "httpx",
    "httpcore",
    "urllib3",
    "hf_xet",
    "filelock",
    "fsspec",
    "aiohttp",
):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# Every OpenImages-backed partition resolves its `image` field against the same
# bucket; the field is the object key's basename.
_OPENIMAGES = "https://s3.amazonaws.com/open-images-dataset/train/{image}"

# Only partitions whose images are reachable from the `image` field alone can be
# downloaded unattended. The value is the URL template.
IMAGE_SOURCES = {
    "captioning_1": _OPENIMAGES,
    "captioning_2": _OPENIMAGES,
    "vqa_1": _OPENIMAGES,
    "vqa_2": _OPENIMAGES,
    "vqa_3": _OPENIMAGES,
}

# Mean bytes per OpenImages train image, measured over the bucket. Used only to
# put a size next to a --dry-run count.
_MEAN_IMAGE_BYTES = 296_000

# OpenImages has removed keys over the years, so this one is expected in small
# numbers and is not worth retrying.
_HTTP_NOT_FOUND = 404


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download externally-hosted images for Llama-Nemotron-VLM-Dataset-v1 "
            "partitions, sampled identically to export_nemotron_vlm.py."
        )
    )
    parser.add_argument(
        "--partitions",
        type=str,
        required=True,
        help=(
            "Comma-separated partitions to download images for. Downloadable: "
            + ", ".join(sorted(IMAGE_SOURCES))
        ),
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help=(
            "Fraction of each partition's rows to fetch images for, 0<f<=1. Must "
            "match the --fraction passed to export_nemotron_vlm.py (default: 1.0)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Sampling seed; must match export_nemotron_vlm.py (default: 0).",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help=(
            "Directory holding the dataset's JSONL files. Defaults to the "
            "HuggingFace cache copy."
        ),
    )
    parser.add_argument(
        "--image-source",
        type=Path,
        default=None,
        help=(
            "Directory to create <partition>_images/ under. Defaults to the "
            "dataset directory, which is where the export looks by default."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=64,
        help="Simultaneous HTTP requests (default: 64).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Attempts per image before giving up (default: 5).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report how many images are missing and their approximate size, then exit."
        ),
    )
    return parser.parse_args()


def select_partitions(dataset_dir: Path, requested: str) -> list[str]:
    """Validate --partitions against what can actually be downloaded."""
    names = [name.strip() for name in requested.split(",") if name.strip()]
    if not names:
        raise ValueError("--partitions was empty")

    missing_jsonl = [n for n in names if not (dataset_dir / f"{n}.jsonl").is_file()]
    if missing_jsonl:
        raise FileNotFoundError(
            f"No JSONL for partition(s) {', '.join(missing_jsonl)} under "
            f"{dataset_dir}. Download the metadata with\n"
            f"    hf download {NEMOTRON_HF_PATH} --repo-type dataset"
        )

    unsupported = [n for n in names if n not in IMAGE_SOURCES]
    if unsupported:
        raise ValueError(
            f"Partition(s) {', '.join(unsupported)} do not have derivable image "
            "URLs, so they cannot be downloaded unattended: their images come "
            "from source datasets that must be obtained by hand. See "
            f"{dataset_dir}/<partition>.md for where each one's images live.\n"
            f"Downloadable partitions: {', '.join(sorted(IMAGE_SOURCES))}"
        )
    return names


def wanted_images(
    dataset_dir: Path, partition: str, fraction: float, seed: int
) -> set[str]:
    """Image names the export will ask for, from the same sampling it uses.

    Streamed rather than materialising the rows: at ``--fraction 1.0`` vqa_1 is
    1.28M rows, and only the image names are needed. Deduplicated because some
    partitions ask several questions about one image.
    """
    names: set[str] = set()
    with (dataset_dir / f"{partition}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("id")
            image = row.get("image")
            if not row_id or not isinstance(image, str) or not image:
                continue
            if keeps_row(partition, str(row_id), fraction, seed):
                names.add(image)
    return names


def already_present(image_dir: Path, names: set[str]) -> set[str]:
    """Names whose file is already downloaded, so a rerun resumes.

    The directory is listed once and intersected, rather than asking about each
    name in turn. At full ``vqa_1`` scale that is the difference between one
    directory sweep and 1.2M individual stat calls, which matters on the
    networked filesystems these runs use: on Lustre each stat is a round trip
    to the metadata server.
    """
    try:
        with os.scandir(image_dir) as entries:
            on_disk = {entry.name for entry in entries if not entry.is_dir()}
    except FileNotFoundError:
        return set()
    return names & on_disk


async def fetch_one(
    session: aiohttp.ClientSession,
    url: str,
    destination: Path,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> tuple[bool, int, str | None]:
    """Fetch one image. Returns (ok, bytes_written, permanent_error).

    A 404 is permanent -- OpenImages has removed images over time -- and is
    reported rather than retried, so a handful of dead keys cannot fail a run
    of a million. Everything else is retried with jittered backoff.
    """
    for attempt in range(max_retries):
        try:
            async with semaphore, session.get(url) as response:
                if response.status == _HTTP_NOT_FOUND:
                    return False, 0, "404"
                response.raise_for_status()
                payload = await response.read()
        # Deliberately broad: every transport failure, timeout and 5xx is worth
        # one more attempt, and a million-image run must not die on the one
        # error class that was not anticipated.
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                return False, 0, type(exc).__name__
            await asyncio.sleep(min(2**attempt, 30) * (0.5 + random.random()))
            continue

        # Written under a temp name so an interrupted run cannot leave a
        # truncated file that the resume check would count as complete.
        partial = destination.with_name(destination.name + ".partial")
        partial.write_bytes(payload)
        partial.replace(destination)
        return True, len(payload), None

    return False, 0, "retries_exhausted"


async def download_partition(
    partition: str,
    image_dir: Path,
    names: list[str],
    concurrency: int,
    max_retries: int,
    timeout: float,
) -> tuple[int, int, dict[str, int]]:
    """Download a partition's missing images. Returns (ok, bytes, failures)."""
    semaphore = asyncio.Semaphore(concurrency)
    failures: dict[str, int] = {}
    num_ok = 0
    total_bytes = 0

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    template = IMAGE_SOURCES[partition]

    async with aiohttp.ClientSession(
        connector=connector, timeout=client_timeout
    ) as session:
        tasks = [
            asyncio.create_task(
                fetch_one(
                    session,
                    template.format(image=name),
                    image_dir / name,
                    semaphore,
                    max_retries,
                )
            )
            for name in names
        ]
        with tqdm(total=len(tasks), desc=f"{partition} images", unit="img") as progress:
            for future in asyncio.as_completed(tasks):
                ok, num_bytes, error = await future
                if ok:
                    num_ok += 1
                    total_bytes += num_bytes
                else:
                    failures[error or "unknown"] = (
                        failures.get(error or "unknown", 0) + 1
                    )
                progress.update(1)
                progress.set_postfix_str(f"{total_bytes / 1e9:.1f} GB")

    return num_ok, total_bytes, failures


def main() -> None:
    args = parse_args()

    if not 0.0 < args.fraction <= 1.0:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")

    dataset_dir = resolve_dataset_dir(args.dataset_path)
    partitions = select_partitions(dataset_dir, args.partitions)
    image_source = args.image_source or dataset_dir

    logger.info("Dataset:      %s", dataset_dir)
    logger.info("Image source: %s", image_source)
    logger.info("Fraction:     %s (seed %d)", args.fraction, args.seed)

    plans = []
    for partition in partitions:
        total = count_rows(dataset_dir / f"{partition}.jsonl")
        names = wanted_images(dataset_dir, partition, args.fraction, args.seed)
        image_dir = partition_image_dir(image_source, partition)
        present = already_present(image_dir, names)
        todo = sorted(names - present)
        plans.append((partition, image_dir, todo))
        logger.info(
            "%s: %d rows -> %d images selected, %d already present, %d to fetch "
            "(~%.1f GB)",
            partition,
            total,
            len(names),
            len(present),
            len(todo),
            len(todo) * _MEAN_IMAGE_BYTES / 1e9,
        )

    num_todo = sum(len(todo) for _, _, todo in plans)
    if args.dry_run:
        logger.info(
            "Dry run: %d images to fetch, roughly %.1f GB. Rerun without "
            "--dry-run to download.",
            num_todo,
            num_todo * _MEAN_IMAGE_BYTES / 1e9,
        )
        return
    if num_todo == 0:
        logger.info("Every selected image is already downloaded; nothing to do.")
        return

    grand_total_ok = 0
    grand_total_bytes = 0
    grand_failures: dict[str, int] = {}
    for partition, image_dir, todo in plans:
        if not todo:
            continue
        image_dir.mkdir(parents=True, exist_ok=True)
        num_ok, num_bytes, failures = asyncio.run(
            download_partition(
                partition,
                image_dir,
                todo,
                args.concurrency,
                args.max_retries,
                args.timeout,
            )
        )
        grand_total_ok += num_ok
        grand_total_bytes += num_bytes
        for reason, count in failures.items():
            grand_failures[reason] = grand_failures.get(reason, 0) + count

    logger.info(
        "Downloaded %d images (%.1f GB) into %s",
        grand_total_ok,
        grand_total_bytes / 1e9,
        image_source,
    )
    if grand_failures:
        detail = ", ".join(
            f"{reason}={count}" for reason, count in sorted(grand_failures.items())
        )
        logger.warning(
            "%d images could not be fetched (%s). Rows referencing them are "
            "skipped by the export, which reports the count.",
            sum(grand_failures.values()),
            detail,
        )
    logger.info(
        "Next: python scripts/export_nemotron_vlm.py --partitions %s "
        "--fraction %s --seed %d%s",
        args.partitions,
        args.fraction,
        args.seed,
        "" if args.image_source is None else f" --image-source {image_source}",
    )


if __name__ == "__main__":
    main()
