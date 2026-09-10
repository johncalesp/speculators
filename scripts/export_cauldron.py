#!/usr/bin/env python3
"""Export Cauldron images and independent questions for on-policy regeneration.

Each HuggingFaceM4/the_cauldron row contains one or more images and a ``texts``
list with independent ``{user, assistant, source}`` annotations. Original
assistant answers are off-policy and are dropped. Every user question becomes
its own conversation while all questions from the row reuse the same
content-addressed image files.
"""

import argparse
import hashlib
import json
import logging
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from datasets import Image, load_dataset
from huggingface_hub import snapshot_download
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
for _noisy in ("httpx", "httpcore", "urllib3", "hf_xet", "filelock", "fsspec"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

CAULDRON_HF_PATH = "HuggingFaceM4/the_cauldron"
CAULDRON_SUBSETS = (
    "ai2d",
    "aokvqa",
    "chart2text",
    "chartqa",
    "clevr",
    "clevr_math",
    "cocoqa",
    "datikz",
    "diagram_image_to_text",
    "docvqa",
    "dvqa",
    "figureqa",
    "finqa",
    "geomverse",
    "hateful_memes",
    "hitab",
    "iam",
    "iconqa",
    "infographic_vqa",
    "intergps",
    "localized_narratives",
    "mapqa",
    "mimic_cgd",
    "multihiertt",
    "nlvr2",
    "ocrvqa",
    "okvqa",
    "plotqa",
    "raven",
    "rendered_text",
    "robut_sqa",
    "robut_wikisql",
    "robut_wtq",
    "scienceqa",
    "screen2words",
    "spot_the_diff",
    "st_vqa",
    "tabmwp",
    "tallyqa",
    "tat_qa",
    "textcaps",
    "textvqa",
    "tqa",
    "vistext",
    "visual7w",
    "visualmrc",
    "vqarad",
    "vqav2",
    "vsr",
    "websight",
)

_MAGIC_SUFFIXES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"\xff\xd8\xff", ".jpg"),
    (b"GIF87a", ".gif"),
    (b"GIF89a", ".gif"),
    (b"BM", ".bmp"),
)
_MAX_IMAGE_SUFFIX_LENGTH = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--subsets",
        help="Comma-separated Cauldron subsets (default: all 50 official subsets)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="One deterministic row fraction applied to every subset (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Local Cauldron snapshot containing <subset>/*.parquet",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help=(
            "Stream selected subsets from Hugging Face instead of requiring "
            "local data"
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        help="Pool identity file (default: cauldron_pool.json next to --outfile)",
    )
    args = parser.parse_args()
    if not 0.0 < args.fraction <= 1.0:
        parser.error("--fraction must be in (0, 1]")
    if args.dataset_path is not None and args.allow_download:
        parser.error("--dataset-path and --allow-download are mutually exclusive")
    if args.pool_manifest is None:
        args.pool_manifest = args.outfile.parent / "cauldron_pool.json"
    return args


def parse_subsets(value: str | None) -> list[str]:
    subsets = (
        list(CAULDRON_SUBSETS)
        if value is None or not value.strip()
        else [item.strip() for item in value.split(",") if item.strip()]
    )
    duplicates = sorted({name for name in subsets if subsets.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate subset(s): {', '.join(duplicates)}")
    unknown = sorted(set(subsets) - set(CAULDRON_SUBSETS))
    if unknown:
        raise ValueError(
            f"Unknown Cauldron subset(s): {', '.join(unknown)}. "
            f"Available: {', '.join(CAULDRON_SUBSETS)}"
        )
    return subsets


def resolve_dataset_dir(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"--dataset-path is not a directory: {resolved}")
        return resolved
    try:
        snapshot = snapshot_download(
            CAULDRON_HF_PATH, repo_type="dataset", local_files_only=True
        )
    except Exception as error:
        raise FileNotFoundError(
            f"{CAULDRON_HF_PATH} is not in the local Hugging Face cache. "
            "Download the selected subset directories with `hf download`, set "
            "--dataset-path, or explicitly pass --allow-download."
        ) from error
    return Path(snapshot)


def subset_rows(
    subset: str, dataset_dir: Path | None, allow_download: bool
) -> Iterator[dict]:
    if allow_download:
        dataset = load_dataset(
            CAULDRON_HF_PATH, subset, split="train", streaming=True
        )
    else:
        if dataset_dir is None:
            raise ValueError("dataset_dir is required for local loading")
        shards = sorted((dataset_dir / subset).glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"No parquet files found for subset '{subset}' under "
                f"{dataset_dir / subset}. Download it with:\n"
                f"  hf download {CAULDRON_HF_PATH} --repo-type dataset "
                f"--include '{subset}/*'"
            )
        dataset = load_dataset(
            "parquet",
            data_files=[str(shard) for shard in shards],
            split="train",
            streaming=True,
        )
    if (dataset.features or {}).get("images") is not None:
        dataset = dataset.cast_column("images", [Image(decode=False)])
    return iter(dataset)


def keeps_row(subset: str, row_index: int, fraction: float, seed: int) -> bool:
    if fraction >= 1.0:
        return True
    digest = hashlib.sha256(f"{seed}:{subset}:{row_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") < fraction * (1 << 64)


def _guess_suffix(data: bytes, source_path: str = "") -> str:
    for magic, guessed in _MAGIC_SUFFIXES:
        if data.startswith(magic):
            return guessed
    if data[8:12] == b"WEBP":
        return ".webp"
    suffix = Path(source_path).suffix.lower()
    if suffix and len(suffix) <= _MAX_IMAGE_SUFFIX_LENGTH:
        return suffix
    return ".img"


def image_bytes(image: Any) -> tuple[bytes, str] | None:
    if not isinstance(image, dict):
        return None
    data = image.get("bytes")
    source_path = str(image.get("path") or "")
    if data:
        return bytes(data), source_path
    path = Path(source_path)
    if path.is_file():
        return path.read_bytes(), source_path
    return None


def write_images(images: Any, image_dir: Path) -> list[Path]:
    if not isinstance(images, list):
        return []
    paths = []
    for image in images:
        loaded = image_bytes(image)
        if loaded is None:
            continue
        data, source_path = loaded
        digest = hashlib.sha256(data).hexdigest()
        destination = image_dir / f"{digest}{_guess_suffix(data, source_path)}"
        if not destination.exists():
            partial = destination.with_name(destination.name + ".partial")
            partial.write_bytes(data)
            partial.replace(destination)
        paths.append(destination.resolve())
    return paths


def build_rows(
    row: dict, subset: str, row_index: int, image_dir: Path
) -> list[dict]:
    image_paths = write_images(row.get("images"), image_dir)
    if not image_paths:
        return []
    exported = []
    texts = row.get("texts")
    if not isinstance(texts, list):
        return exported
    for question_index, annotation in enumerate(texts):
        if not isinstance(annotation, dict):
            continue
        user = annotation.get("user")
        if not isinstance(user, str) or not user.strip():
            continue
        content = [
            {"type": "image", "path": str(path)} for path in image_paths
        ] + [{"type": "text", "text": user.strip()}]
        exported.append(
            {
                "conversation_id": (
                    f"cauldron/{subset}/{row_index}/{question_index}"
                ),
                "conversations": [{"role": "user", "content": content}],
                "metadata": {
                    "dataset": CAULDRON_HF_PATH,
                    "subset": subset,
                    "source": annotation.get("source"),
                },
            }
        )
    return exported


def load_exported_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if conversation_id := row.get("conversation_id"):
                ids.add(str(conversation_id))
    return ids


def validate_pool_manifest(
    path: Path,
    outfile: Path,
    subsets: list[str],
    fraction: float,
    seed: int,
) -> None:
    desired = {
        "dataset": CAULDRON_HF_PATH,
        "subsets": subsets,
        "fraction": fraction,
        "seed": seed,
        "question_mode": "independent",
    }
    if path.is_file():
        existing = json.loads(path.read_text())
        identity_keys = ("dataset", "subsets", "seed", "question_mode")
        changed = [key for key in identity_keys if existing.get(key) != desired[key]]
        if changed:
            raise ValueError(
                "Cauldron pool identity changed "
                f"({', '.join(changed)}). Use a new output directory."
            )
        previous_fraction = float(existing.get("fraction", 0))
        if fraction < previous_fraction:
            raise ValueError(
                f"Cannot shrink PERC_SAMPLES from {previous_fraction:g} to "
                f"{fraction:g} in an append-only pool. Keep the larger pool and "
                "use MAX_SAMPLES for a smaller training run."
            )
    elif outfile.is_file() and outfile.stat().st_size:
        raise ValueError(
            f"{outfile} exists without {path}; refusing to append to an "
            "unidentified conversation pool."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(desired, indent=2) + "\n")
    temporary.replace(path)


def update_pool_count(path: Path, conversation_count: int) -> None:
    """Atomically record the count only after a completed export."""
    manifest = json.loads(path.read_text())
    manifest["conversation_count"] = conversation_count
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def format_sample_cap_error(
    current_fraction: float, available: int, requested: int
) -> str:
    """Explain how much a sampled pool must grow to satisfy a training cap."""
    lines = [
        f"MAX_SAMPLES={requested:,} exceeds the reusable Cauldron pool of "
        f"{available:,} conversations at PERC_SAMPLES={current_fraction:g}."
    ]
    if available <= 0:
        lines.append("The pool is empty; verify the selected subsets and image data.")
        return "\n".join(lines)

    required_raw = current_fraction * requested / available
    estimated_full = math.floor(available / current_fraction)
    if required_raw > 1:
        lines.append(
            "Even PERC_SAMPLES=1 is estimated to provide only "
            f"{estimated_full:,} conversations, so 100% is insufficient. "
            "Select more subsets or lower MAX_SAMPLES."
        )
        return "\n".join(lines)

    required = math.ceil(required_raw * 1000) / 1000
    increase = required - current_fraction
    relative = increase / current_fraction * 100
    lines.append(
        f"Increase PERC_SAMPLES from {current_fraction:g} to an estimated minimum "
        f"of {required:g} (+{increase:g}, {relative:.1f}% relative), then rerun. "
        "A slightly larger value allows for deterministic-sampling variation."
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    subsets = parse_subsets(args.subsets)
    dataset_dir = (
        None if args.allow_download else resolve_dataset_dir(args.dataset_path)
    )
    validate_pool_manifest(
        args.pool_manifest,
        args.outfile,
        subsets,
        args.fraction,
        args.seed,
    )
    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.outfile.parent.mkdir(parents=True, exist_ok=True)
    exported_ids = load_exported_ids(args.outfile) if args.resume else set()

    mode = "Hugging Face streaming" if args.allow_download else str(dataset_dir)
    logger.info("Cauldron source: %s", mode)
    logger.info(
        "Subsets: %d; PERC_SAMPLES=%.4g; existing conversations=%d",
        len(subsets),
        args.fraction,
        len(exported_ids),
    )
    written = skipped = 0
    with (
        args.outfile.open("a" if args.resume else "w", encoding="utf-8") as handle,
        tqdm(desc="Exporting Cauldron", unit="row") as progress,
    ):
        for subset in subsets:
            subset_written = 0
            for row_index, row in enumerate(
                subset_rows(subset, dataset_dir, args.allow_download)
            ):
                progress.update(1)
                if not keeps_row(subset, row_index, args.fraction, args.seed):
                    continue
                rows = build_rows(row, subset, row_index, args.image_dir)
                if not rows:
                    skipped += 1
                    continue
                for exported in rows:
                    if exported["conversation_id"] in exported_ids:
                        continue
                    handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
                    exported_ids.add(exported["conversation_id"])
                    written += 1
                    subset_written += 1
            handle.flush()
            logger.info("%-24s added %d conversations", subset, subset_written)

    logger.info(
        "Added %d conversations (%d total, %d selected rows unusable)",
        written,
        len(exported_ids),
        skipped,
    )
    update_pool_count(args.pool_manifest, len(exported_ids))


if __name__ == "__main__":
    main()
