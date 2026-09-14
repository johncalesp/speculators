#!/usr/bin/env python3
"""Select a balanced, image-disjoint Cauldron train/validation dataset.

The exporter intentionally keeps a reusable superset of regenerated
conversations. This script derives a cheap, reproducible training view from
that pool: it filters a task profile, limits correlated questions per image,
balances capped selections across subsets, and assigns every image group to
exactly one split.
"""

import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

LLAVA_WILD_SUBSETS = (
    "aokvqa",
    "cocoqa",
    "localized_narratives",
    "nlvr2",
    "ocrvqa",
    "okvqa",
    "screen2words",
    "spot_the_diff",
    "st_vqa",
    "tallyqa",
    "textcaps",
    "textvqa",
    "visual7w",
    "vqav2",
    "vsr",
)

PROFILES: dict[str, tuple[str, ...] | None] = {
    "llava_wild": LLAVA_WILD_SUBSETS,
    "all": None,
}
_CONVERSATION_ID_PARTS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="llava_wild",
        help="Built-in subset profile (default: llava_wild)",
    )
    parser.add_argument(
        "--subsets",
        help="Comma-separated subset override; takes precedence over --profile",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Total train plus validation conversations (default: all eligible)",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-questions-per-image", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.max_samples is not None and args.max_samples <= 1:
        parser.error("--max-samples must be greater than 1")
    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.max_questions_per_image <= 0:
        parser.error("--max-questions-per-image must be positive")
    if args.manifest is None:
        args.manifest = args.outfile.with_suffix(".manifest.json")
    return args


def stable_hash(value: str, seed: int, purpose: str) -> int:
    digest = hashlib.sha256(f"{seed}:{purpose}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def parse_conversation_id(conversation_id: str) -> tuple[str, str] | None:
    parts = conversation_id.split("/")
    if (
        len(parts) != _CONVERSATION_ID_PARTS
        or parts[0] != "cauldron"
        or not parts[2].isdigit()
        or not parts[3].isdigit()
    ):
        return None
    return parts[1], "/".join(parts[:3])


def image_keys(row: dict[str, Any]) -> tuple[str, ...]:
    keys: set[str] = set()
    conversations = row.get("conversations")
    if not isinstance(conversations, list):
        return ()
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        content = turn.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {
                "image",
                "image_url",
            }:
                continue
            value = part.get("path")
            if value is None and isinstance(part.get("image_url"), dict):
                value = part["image_url"].get("url")
            if isinstance(value, str) and value:
                keys.add(Path(value.removeprefix("file://")).name)
    return tuple(sorted(keys))


def group_id_for(row: dict[str, Any], source_group: str) -> str:
    images = image_keys(row)
    if not images:
        return source_group
    digest = hashlib.sha256("\0".join(images).encode()).hexdigest()
    return f"cauldron/images/{digest}"


def selected_subsets(profile: str, override: str | None) -> set[str] | None:
    if override is not None:
        selected = {item.strip() for item in override.split(",") if item.strip()}
        if not selected:
            raise ValueError("--subsets did not contain any subset names")
        return selected
    configured = PROFILES[profile]
    return None if configured is None else set(configured)


def load_candidates(
    path: Path,
    subsets: set[str] | None,
    *,
    seed: int,
    max_questions_per_image: int,
) -> tuple[dict[str, tuple[str, str]], dict[str, list[tuple[int, str]]], int]:
    candidate_info: dict[str, tuple[str, str]] = {}
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    malformed = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            conversation_id = str(row.get("conversation_id") or "")
            parsed = parse_conversation_id(conversation_id)
            if parsed is None:
                malformed += 1
                continue
            subset, source_group = parsed
            if subsets is not None and subset not in subsets:
                continue
            group_id = group_id_for(row, source_group)
            candidate_info[conversation_id] = (subset, group_id)
            grouped[group_id].append(
                (
                    stable_hash(conversation_id, seed, "question"),
                    conversation_id,
                    subset,
                )
            )

    by_split: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for candidates in grouped.values():
        for _, conversation_id, subset in sorted(candidates)[:max_questions_per_image]:
            by_split[subset].append(
                (stable_hash(conversation_id, seed, "selection"), conversation_id)
            )
    return candidate_info, by_split, malformed


def partition_candidates(
    candidate_info: dict[str, tuple[str, str]],
    candidates: dict[str, list[tuple[int, str]]],
    *,
    seed: int,
    val_fraction: float,
) -> tuple[dict[str, list[tuple[int, str]]], dict[str, list[tuple[int, str]]]]:
    train: dict[str, list[tuple[int, str]]] = defaultdict(list)
    val: dict[str, list[tuple[int, str]]] = defaultdict(list)
    threshold = val_fraction * (1 << 64)
    for subset, items in candidates.items():
        for rank, conversation_id in items:
            info = candidate_info.get(conversation_id)
            if info is None:
                raise ValueError(
                    f"Candidate metadata is missing for: {conversation_id}"
                )
            group_id = info[1]
            target = val if stable_hash(group_id, seed, "split") < threshold else train
            target[subset].append((rank, conversation_id))
    return train, val


def balanced_take(
    candidates: dict[str, list[tuple[int, str]]], limit: int | None
) -> list[str]:
    queues = {
        subset: deque(conversation_id for _, conversation_id in sorted(items))
        for subset, items in sorted(candidates.items())
        if items
    }
    available = sum(len(items) for items in queues.values())
    target = available if limit is None else limit
    if target > available:
        raise ValueError(
            f"Requested {target:,} samples but only {available:,} are eligible"
        )
    selected: list[str] = []
    active = deque(queues)
    while active and len(selected) < target:
        subset = active.popleft()
        selected.append(queues[subset].popleft())
        if queues[subset]:
            active.append(subset)
    return selected


def split_targets(
    requested: int,
    val_fraction: float,
    available_train: int,
    available_val: int,
) -> tuple[int, int]:
    if available_train == 0 or available_val == 0:
        raise ValueError(
            "The deterministic split produced an empty train or validation pool; "
            "increase PERC_SAMPLES or change EXPORT_SEED."
        )
    val_target = max(1, round(requested * val_fraction))
    val_target = min(val_target, available_val)
    train_target = requested - val_target
    if train_target > available_train:
        train_target = available_train
        val_target = requested - train_target
    if val_target > available_val or train_target <= 0:
        raise ValueError(
            "Not enough eligible rows to satisfy both train and validation splits"
        )
    return train_target, val_target


def write_selection(
    source: Path,
    candidate_info: dict[str, tuple[str, str]],
    train_ids: list[str],
    val_ids: list[str],
    outfile: Path,
) -> None:
    selected = dict.fromkeys(train_ids, "train")
    selected.update(dict.fromkeys(val_ids, "val"))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    temporary = outfile.with_name(outfile.name + ".tmp")
    written: set[str] = set()
    with (
        source.open(encoding="utf-8") as input_handle,
        temporary.open("w", encoding="utf-8") as handle,
    ):
        for line in input_handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            conversation_id = str(row.get("conversation_id") or "")
            split = selected.get(conversation_id)
            if split is not None:
                subset, group_id = candidate_info[conversation_id]
                output = dict(row)
                output["subset"] = subset
                output["group_id"] = group_id
                output["data_split"] = split
                handle.write(json.dumps(output, ensure_ascii=False) + "\n")
                written.add(conversation_id)
    missing = set(selected) - written
    if missing:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"{len(missing)} selected conversations disappeared while rereading "
            f"{source}"
        )
    temporary.replace(outfile)


def selection_digest(train_ids: list[str], val_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for split, conversation_ids in (("train", train_ids), ("val", val_ids)):
        for conversation_id in sorted(conversation_ids):
            digest.update(f"{split}\0{conversation_id}\n".encode())
    return digest.hexdigest()


def subset_counts(conversation_ids: list[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for conversation_id in conversation_ids:
        parsed = parse_conversation_id(conversation_id)
        if parsed is not None:
            counts[parsed[0]] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    subsets = selected_subsets(args.profile, args.subsets)
    candidate_info, candidates, malformed = load_candidates(
        args.data,
        subsets,
        seed=args.seed,
        max_questions_per_image=args.max_questions_per_image,
    )
    train_candidates, val_candidates = partition_candidates(
        candidate_info,
        candidates,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    available_train = sum(map(len, train_candidates.values()))
    available_val = sum(map(len, val_candidates.values()))
    available = available_train + available_val
    requested = available if args.max_samples is None else args.max_samples
    if requested > available:
        raise ValueError(
            f"MAX_SAMPLES={requested:,} exceeds the selected Cauldron pool of "
            f"{available:,} conversations after profile and image-question caps. "
            "Increase PERC_SAMPLES, select more subsets, raise "
            "MAX_QUESTIONS_PER_IMAGE, or lower MAX_SAMPLES."
        )
    train_target, val_target = split_targets(
        requested,
        args.val_fraction,
        available_train,
        available_val,
    )
    train_ids = balanced_take(train_candidates, train_target)
    val_ids = balanced_take(val_candidates, val_target)
    write_selection(
        args.data,
        candidate_info,
        train_ids,
        val_ids,
        args.outfile,
    )

    manifest = {
        "source": str(args.data.resolve()),
        "profile": args.profile,
        "subsets": sorted(subsets) if subsets is not None else "all",
        "seed": args.seed,
        "max_samples": args.max_samples,
        "val_fraction": args.val_fraction,
        "max_questions_per_image": args.max_questions_per_image,
        "available_after_caps": available,
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "train_by_subset": subset_counts(train_ids),
        "val_by_subset": subset_counts(val_ids),
        "selection_sha256": selection_digest(train_ids, val_ids),
        "malformed_rows": malformed,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(args.manifest.name + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.manifest)
    print(
        f"Selected {len(train_ids):,} train and {len(val_ids):,} validation "
        f"conversations from {len(candidates)} subsets"
    )


if __name__ == "__main__":
    main()
