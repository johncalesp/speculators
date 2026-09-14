"""Recover CLEVR-Math's missing images from the same cached Cauldron snapshot."""

import re
import shutil
from functools import lru_cache
from pathlib import Path


def export_image(value: dict, destination: Path, task: dict) -> bool:
    """Publish one image atomically and report whether CLEVR recovery was needed."""
    if destination.exists():
        return False
    temporary = destination.with_suffix(".tmp")
    recovered = False
    if value.get("bytes") is not None:
        temporary.write_bytes(value["bytes"])
    elif value.get("path"):
        original = Path(value["path"])
        if not original.is_absolute():
            original = Path(task["file"]).parent / original
        if original.is_file():
            shutil.copyfile(original, temporary)
        else:
            temporary.write_bytes(recover_clevr_image(task, original))
            recovered = True
    else:
        raise ValueError(f"{destination.name}: image has neither bytes nor path")
    temporary.replace(destination)
    return recovered


@lru_cache(maxsize=1)
def clevr_index(source: Path):
    """Index filenames, without loading the corpus's embedded image bytes."""
    import pyarrow.parquet as pq

    files = sorted((source / "clevr").glob("*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"CLEVR-Math contains missing external image paths. Cache the clevr "
            f"Parquet subset alongside clevr_math under {source} and retry."
        )
    print(f"Indexing cached CLEVR image filenames from {len(files)} shards", flush=True)
    index = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            rows = parquet.read_row_group(
                group, columns=["images.list.element.path"]
            ).to_pylist()
            for row_index, row in enumerate(rows):
                for image_index, image in enumerate(row["images"]):
                    name = Path(image.get("path") or "").name
                    if not re.fullmatch(r"CLEVR_train_\d{6}\.png", name):
                        continue
                    if name in index:
                        raise ValueError(f"Ambiguous CLEVR image filename: {name}")
                    index[name] = (path, group, row_index, image_index)
    print(f"Indexed {len(index)} CLEVR image filenames", flush=True)
    return index


@lru_cache(maxsize=1)
def clevr_image_group(path: Path, group: int):
    """Retain only one image row group while chunks interleave other subsets."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).read_row_group(group, columns=["images"])


def recover_clevr_image(task: dict, original: Path) -> bytes:
    """Match the explicit CLEVR filename; row orders may differ between subsets."""
    if task["subset"] != "clevr_math" or not re.fullmatch(
        r"CLEVR_train_\d{6}\.png", original.name
    ):
        raise FileNotFoundError(
            f"{task['subset']} chunk {task['id']}: image has no embedded bytes "
            f"and its path does not exist: {original}"
        )
    source = Path(task["file"]).parent.parent
    location = clevr_index(source).get(original.name)
    if location is None:
        raise FileNotFoundError(
            f"{task['subset']} chunk {task['id']}: cannot recover {original.name} "
            f"from {source / 'clevr'}. Cache the complete clevr subset and retry."
        )
    path, group, row_index, image_index = location
    table = clevr_image_group(path, group)
    image = table.slice(row_index, 1).to_pylist()[0]["images"][image_index]
    if Path(image.get("path") or "").name != original.name:
        raise ValueError(f"Cached CLEVR image index changed for {original.name}")
    content = image.get("bytes")
    if not content:
        raise ValueError(
            f"Cached CLEVR donor {original.name} in {path} has no image bytes"
        )
    return content
