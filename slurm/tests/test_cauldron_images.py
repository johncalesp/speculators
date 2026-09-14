from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import cauldron_data as data
import cauldron_images as images


def write_shard(source, subset, values):
    path = source / subset / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "images": [value],
                    "texts": [{"user": "How many objects?", "assistant": "2"}],
                }
                for value in values
            ]
        ),
        path,
        row_group_size=2,
    )
    return path


def clevr_fixture(source):
    write_shard(
        source,
        "clevr",
        [
            {"bytes": b"image-one", "path": "CLEVR_train_000001.png"},
            {"bytes": b"image-zero", "path": "CLEVR_train_000000.png"},
        ],
    )
    write_shard(
        source,
        "clevr_math",
        [
            {
                "bytes": None,
                "path": "/fsx/unavailable/CLEVR_v1.0/images/train/"
                f"CLEVR_train_{index:06d}.png",
            }
            for index in range(2)
        ],
    )
    return data.make_plan(source, "clevr_math", 0, 2)[0]


def exported_images(rows):
    return [
        Path(row["conversations"][0]["content"][0]["path"]).read_bytes() for row in rows
    ]


def test_recovers_by_filename_when_donor_row_order_differs(tmp_path, capsys):
    task = clevr_fixture(tmp_path / "source")
    rows = data.export_chunk(task, tmp_path / "output")
    assert exported_images(rows) == [b"image-zero", b"image-one"]
    assert "recovered 2 CLEVR-Math images" in capsys.readouterr().out
    assert rows[0]["conversations"][1]["content"][0]["text"] == "2"


def test_recovers_after_partial_export_without_rewriting_images(tmp_path, monkeypatch):
    task = clevr_fixture(tmp_path / "source")
    output = tmp_path / "output"
    recover = images.recover_clevr_image

    def interrupted(task, original):
        if original.name == "CLEVR_train_000001.png":
            raise OSError("Interrupted export")
        return recover(task, original)

    monkeypatch.setattr(images, "recover_clevr_image", interrupted)
    with pytest.raises(OSError, match="Interrupted export"):
        data.export_chunk(task, output)
    chunk = output / "chunks" / task["id"]
    first = chunk / f"{task['id']}-0000-0.image"
    modified = first.stat().st_mtime_ns
    assert not (chunk / "conversations.json").exists()
    monkeypatch.setattr(images, "recover_clevr_image", recover)
    images.clevr_index.cache_clear()
    images.clevr_image_group.cache_clear()
    assert exported_images(data.export_chunk(task, output)) == [
        b"image-zero",
        b"image-one",
    ]
    assert first.stat().st_mtime_ns == modified


@pytest.mark.parametrize(
    "kind", ["missing_subset", "missing_name", "duplicate", "empty"]
)
def test_missing_or_ambiguous_donors_fail_without_publishing_chunk(tmp_path, kind):
    source = tmp_path / "source"
    task = clevr_fixture(source)
    if kind == "missing_subset":
        next((source / "clevr").glob("*.parquet")).unlink()
    else:
        values = [{"bytes": b"image", "path": "CLEVR_train_000000.png"}]
        if kind == "duplicate":
            values *= 2
        elif kind == "empty":
            values[0]["bytes"] = None
        write_shard(source, "clevr", values)
    output = tmp_path / "output"
    with pytest.raises((FileNotFoundError, ValueError), match="CLEVR|clevr"):
        data.export_chunk(task, output)
    assert not (output / "chunks" / task["id"] / "conversations.json").exists()


def test_other_subsets_do_not_use_clevr_filename_fallback(tmp_path):
    task = clevr_fixture(tmp_path / "source")
    task["subset"] = "unrelated"
    with pytest.raises(FileNotFoundError, match="unrelated chunk"):
        data.export_chunk(task, tmp_path / "output")


def test_existing_local_paths_and_embedded_bytes_take_precedence(tmp_path):
    local = tmp_path / "CLEVR_train_000000.png"
    local.write_bytes(b"local-image")
    source = tmp_path / "source"
    write_shard(
        source,
        "clevr_math",
        [
            {"bytes": None, "path": str(local)},
            {"bytes": b"embedded-image", "path": str(local)},
        ],
    )
    task = data.make_plan(source, "clevr_math", 0, 2)[0]
    assert exported_images(data.export_chunk(task, tmp_path / "output")) == [
        b"local-image",
        b"embedded-image",
    ]
