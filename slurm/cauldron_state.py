"""Filesystem transactions shared by the Cauldron workers."""

import json
import os
import shutil
from pathlib import Path


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_journal(path: Path) -> list[dict]:
    """Recover a torn final append; reject corruption in committed lines."""
    if not path.exists():
        return []
    records = []
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                stream.truncate(start)
                break
            records.append(json.loads(line))
    return records


def append_journal(path: Path, record: dict) -> None:
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def publish_checkpoint(root: Path, epoch: int) -> None:
    """Keep the last complete checkpoint until its replacement is complete."""
    destination = root / str(epoch)
    previous = root / f".{epoch}.previous"
    staging = root / f".{epoch}.incomplete"
    if previous.exists():
        shutil.rmtree(previous)
    if destination.exists():
        destination.rename(previous)
    staging.rename(destination)
    if previous.exists():
        shutil.rmtree(previous)


def recover_checkpoints(root: Path) -> None:
    """Recover a crash between the two renames; discard unpublished writes."""
    root.mkdir(parents=True, exist_ok=True)
    for previous in root.glob(".*.previous"):
        epoch = previous.name.split(".")[1]
        if not epoch.isdecimal():
            continue
        destination = root / epoch
        if destination.exists():
            shutil.rmtree(previous)
        else:
            previous.rename(destination)
    for staging in root.glob(".*.incomplete"):
        if staging.name.split(".")[1].isdecimal():
            shutil.rmtree(staging)


def training_complete(root: Path, epochs: int) -> bool:
    state = root / str(epochs - 1) / "training_state.json"
    if not state.is_file():
        return False
    value = json.loads(state.read_text())
    return value["epoch"] == epochs - 1 and value["local_step"] == 0
