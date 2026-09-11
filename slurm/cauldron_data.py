"""Resumable Cauldron image export, on-policy regeneration, and Arrow shards."""

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

from cauldron_state import append_journal, atomic_json, read_journal


def _requested_subsets(source: Path, subsets: str):
    import yaml

    available = {
        p.name for p in source.iterdir() if p.is_dir() and list(p.glob("*.parquet"))
    }
    requested = (
        set(subsets.replace(",", " ").split()) if subsets != "all" else available
    )
    # A partial Hub snapshot must not silently turn "all" into a smaller run.
    card = source / "README.md"
    if subsets == "all":
        if not card.exists():
            raise ValueError(
                "For all subsets, cache README.md too, or specify CAULDRON_SUBSETS"
            )
        metadata = yaml.safe_load(card.read_text().split("---", 2)[1])
        declared = {item["config_name"] for item in metadata.get("configs", [])}
        if not declared:
            declared = {
                item["config_name"] for item in metadata.get("dataset_info", [])
            }
        if not declared:
            raise ValueError("Cannot verify all subsets from the cached dataset card")
        requested = declared
    if not requested or requested - available:
        raise ValueError(
            f"Missing cached Cauldron subsets: {sorted(requested - available)}"
        )
    return requested


def _subset_tasks(source: Path, subset: str, chunk_size: int):
    import pyarrow.parquet as pq

    tasks = []
    files = sorted((source / subset).glob("*.parquet"))
    totals = {int(p.name.split("-of-")[1].split("-")[0].split(".")[0]) for p in files}
    if len(totals) != 1 or len(files) != next(iter(totals)):
        raise ValueError(
            f"Incomplete Parquet shards for {subset}: {len(files)}, expected {totals}"
        )
    for path in files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            count = parquet.metadata.row_group(group).num_rows
            for start in range(0, count, chunk_size):
                tasks.append(
                    {
                        "file": str(path),
                        "subset": subset,
                        "row_group": group,
                        "start": start,
                        "count": min(chunk_size, count - start),
                    }
                )
    return tasks


def make_plan(
    source: Path, subsets: str, max_samples: int, chunk_size: int
) -> list[dict]:
    by_subset = [
        _subset_tasks(source, subset, chunk_size)
        for subset in sorted(_requested_subsets(source, subsets))
    ]
    plan = []
    remaining = max_samples or None
    # Interleave subsets, so limited smoke runs do not only select ai2d.
    for stripe in itertools.zip_longest(*by_subset):
        for task in stripe:
            if task is None:
                continue
            if remaining is not None:
                task["count"] = min(task["count"], remaining)
            task["id"] = f"{len(plan):07d}"
            plan.append(task)
            if remaining is not None:
                remaining -= task["count"]
                if remaining == 0:
                    return plan
    return plan


def export_chunk(task: dict, root: Path) -> list[dict]:
    import pyarrow.parquet as pq

    directory = root / "chunks" / task["id"]
    directory.mkdir(parents=True, exist_ok=True)
    exported = directory / "conversations.json"
    if exported.exists():
        return json.loads(exported.read_text())
    parquet = pq.ParquetFile(task["file"])
    table = parquet.read_row_group(task["row_group"], columns=["images", "texts"])
    rows = table.slice(task["start"], task["count"]).to_pylist()
    conversations = []
    for index, row in enumerate(rows):
        row_id = f"{task['id']}-{index:04d}"
        images = []
        for image_index, value in enumerate(row["images"]):
            image = directory / f"{row_id}-{image_index}.image"
            if not image.exists():
                temporary = image.with_suffix(".tmp")
                if value.get("bytes") is not None:
                    temporary.write_bytes(value["bytes"])
                elif value.get("path"):
                    original = Path(value["path"])
                    if not original.is_absolute():
                        original = Path(task["file"]).parent / original
                    shutil.copyfile(original, temporary)
                else:
                    raise ValueError(f"{row_id}: image has neither bytes nor path")
                temporary.replace(image)
            images.append({"type": "image", "path": str(image.absolute())})
        turns = []
        for turn_index, turn in enumerate(row["texts"]):
            user = turn.get("user")
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f"{row_id}: missing user text at turn {turn_index}")
            content = (
                [*images, {"type": "text", "text": user}]
                if turn_index == 0
                else [{"type": "text", "text": user}]
            )
            turns.append({"role": "user", "content": content})
            # Keep source answers in the exported artifact for inspection only.
            turns.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": turn.get("assistant") or ""}],
                }
            )
        conversations.append(
            {"id": row_id, "conversations": turns, "num_images": len(images)}
        )
    atomic_json(exported, conversations)
    return conversations


def post_chat(endpoint: str, payload: dict) -> dict:
    for attempt in range(5):
        request = urllib.request.Request(
            endpoint + "/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in (408, 429) and error.code < 500:
                raise RuntimeError(
                    f"vLLM HTTP {error.code}: {error.read().decode()[:1000]}"
                ) from error
            if attempt == 4:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt == 4:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("Chat retry budget exhausted")


def regenerate(row: dict, config: dict, endpoint: str) -> dict:
    from speculators.data_generation.preprocessing import (
        _adapt_conv_for_vllm,
        _render_boundary_rows,
    )
    from speculators.data_generation.render_client import render_conversation

    result = {"id": row["id"], "conversations": [], "samples": [], "skipped": None}
    if row["num_images"] > config["max_images"]:
        result["skipped"] = "image_limit"
        return result
    history = []
    for turn in row["conversations"]:
        if turn["role"] != "user":
            continue
        history.append(turn)
        prompt = render_conversation(
            endpoint,
            _adapt_conv_for_vllm(history),
            add_generation_prompt=True,
            truncate_prompt_tokens=config["seq_length"] + 1,
            truncation_side="right",
        )
        available = config["seq_length"] - len(prompt) - 2
        if available <= 0:
            history.pop()
            result["skipped"] = "context_limit"
            break
        response = post_chat(
            endpoint,
            {
                "model": config["model"],
                "messages": _adapt_conv_for_vllm(history),
                "max_tokens": min(config["max_new_tokens"], available),
                "temperature": config["temperature"],
                "seed": config["seed"],
            },
        )
        choice = response["choices"][0]
        content = choice["message"].get("content")
        if choice["finish_reason"] == "length":
            # Do not teach an artificial EOS after a truncated generation.
            history.pop()
            result["skipped"] = "generation_limit"
            break
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"{row['id']}: empty assistant completion")
        history.append(
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        )
    result["conversations"] = history
    # Use the same rendering boundary logic as prepare-data, with images retained.
    for boundary in _render_boundary_rows(history, endpoint, config["seq_length"] + 1):
        if len(boundary["input_ids"]) > config["seq_length"]:
            result["skipped"] = "rendered_context_limit"
            continue
        if not any(boundary["loss_mask"]):
            continue
        result["samples"].append(
            {
                "input_ids": boundary["input_ids"],
                "loss_mask": boundary["loss_mask"],
                "seq_len": len(boundary["input_ids"]),
                "messages": _adapt_conv_for_vllm(boundary["conv"]),
            }
        )
    return result


def training_features():
    from datasets import Features, List, Value

    return Features(
        {
            "input_ids": List(Value("int64")),
            "loss_mask": List(Value("int64")),
            "seq_len": Value("int64"),
            "messages": List(
                {
                    "role": Value("string"),
                    "content": List(
                        {
                            "type": Value("string"),
                            "text": Value("string"),
                            "image_url": {"url": Value("string")},
                        }
                    ),
                }
            ),
        }
    )


def prepare_chunk(task: dict, config: dict, root: Path, endpoint: str, stop: Path):
    directory = root / "chunks" / task["id"]
    complete = directory / "complete.json"
    if complete.exists():
        return
    conversations = export_chunk(task, root)
    journal = directory / "regenerated.jsonl"
    records = read_journal(journal)
    seen = {record["id"] for record in records}
    pending_rows = iter(row for row in conversations if row["id"] not in seen)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config["concurrency"]
    ) as pool:
        pending = {}
        while True:
            while len(pending) < config["concurrency"] and not stop.exists():
                row = next(pending_rows, None)
                if row is None:
                    break
                pending[pool.submit(regenerate, row, config, endpoint)] = row["id"]
            if not pending:
                break
            finished, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in finished:
                row_id = pending.pop(future)
                try:
                    record = future.result()
                except Exception as error:
                    raise RuntimeError(
                        f"Chunk {task['id']}, conversation {row_id} failed"
                    ) from error
                append_journal(journal, record)
                records.append(record)
    if len(records) != len(conversations):
        return
    _finish_chunk(directory, records, config["seed"])


def _finish_chunk(directory, records, seed):
    from datasets import Dataset

    complete = directory / "complete.json"
    records.sort(key=lambda record: record["id"])
    samples = [sample for record in records for sample in record["samples"]]
    random.Random(seed).shuffle(samples)
    if samples:
        temporary = directory / "prepared.incomplete"
        if temporary.exists():
            shutil.rmtree(temporary)
        if not (directory / "prepared").exists():
            dataset = Dataset.from_list(
                samples, features=training_features()
            ).with_format("torch")
            dataset.save_to_disk(str(temporary))
            temporary.rename(directory / "prepared")
    skipped = {}
    for record in records:
        if record["skipped"]:
            reason = record["skipped"]
            skipped[reason] = skipped.get(reason, 0) + 1
    atomic_json(
        complete,
        {"conversations": len(records), "samples": len(samples), "skipped": skipped},
    )
    print(
        f"Prepared {directory.name}: {len(records)} conversations, "
        f"{len(samples)} training rows, skips={skipped}",
        flush=True,
    )


def assemble(root: Path, plan: list[dict], seed: int) -> None:
    """Link immutable Arrow shards; avoid copying the full corpus on each job."""
    from datasets import load_from_disk

    destination = root / "prepared"
    if destination.exists():
        return
    temporary = root / "prepared.incomplete"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    tasks = list(plan)
    random.Random(seed).shuffle(tasks)
    info = state = None
    data_files = []
    total = 0
    for task in tasks:
        directory = root / "chunks" / task["id"]
        report = json.loads((directory / "complete.json").read_text())
        if report["samples"] == 0:
            continue
        shard = directory / "prepared"
        shard_state = json.loads((shard / "state.json").read_text())
        shard_info = json.loads((shard / "dataset_info.json").read_text())
        if info is None:
            info, state = shard_info, shard_state
        elif info["features"] != shard_info["features"]:
            raise ValueError("Prepared shard schemas differ")
        for entry in shard_state["_data_files"]:
            name = task["id"] + "-" + entry["filename"]
            (temporary / name).symlink_to((shard / entry["filename"]).absolute())
            data_files.append({"filename": name})
        total += report["samples"]
    if total < 2:
        raise ValueError(
            "Need at least two prepared training rows; inspect chunk skip reports"
        )
    state["_data_files"] = data_files
    state["_fingerprint"] = hashlib.sha256(json.dumps(data_files).encode()).hexdigest()[
        :16
    ]
    atomic_json(temporary / "state.json", state)
    atomic_json(temporary / "dataset_info.json", info)
    # Validate the composed HF format before publishing it.
    if len(load_from_disk(str(temporary))) != total:
        raise ValueError("Assembled Arrow row count does not match completed shards")
    temporary.rename(destination)
    atomic_json(
        root / "data_complete.json", {"training_rows": total, "chunks": len(plan)}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.output / "pipeline_config.json").read_text())
    plan = json.loads((args.output / "plan.json").read_text())
    for task in plan:
        if args.stop_file.exists():
            return
        prepare_chunk(task, config, args.output, args.endpoint, args.stop_file)
    if not args.stop_file.exists():
        assemble(args.output, plan, config["seed"])


if __name__ == "__main__":
    main()
