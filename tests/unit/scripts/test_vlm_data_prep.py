"""Dependency-light tests for the multimodal (VLM) data-prep scripts.

No network and no HF download: ``export_visionarena.py``'s row handling is
exercised against the real VisionArena row shape, and
``regenerate_vlm_responses.py``'s regeneration loop is driven over a fake
endpoint.

The most important assertions here are the handoff ones: what regeneration
writes has to be exactly what ``prepare_data.py`` accepts. Its real adapter,
``_adapt_conv_for_vllm``, is used to check both that the output form is accepted
and that the wire form is not -- the reason the script keeps two views of the
same conversation.

The scripts are not packages, so they are imported by path.
"""

import asyncio
import importlib.util
import io
import json
import struct
import tarfile
import zlib
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset, Features, Image, Value

from speculators.data_generation.preprocessing import _adapt_conv_for_vllm

_SCRIPTS_DIR = Path(__file__).resolve().parents[3] / "scripts"


def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export = _load_script("export_visionarena", _SCRIPTS_DIR / "export_visionarena.py")
nemotron = _load_script("export_nemotron_vlm", _SCRIPTS_DIR / "export_nemotron_vlm.py")
regen = _load_script(
    "regenerate_vlm_responses", _SCRIPTS_DIR / "regenerate_vlm_responses.py"
)
download = _load_script(
    "download_nemotron_images", _SCRIPTS_DIR / "download_nemotron_images.py"
)


# A minimal but real PNG header, enough for suffix sniffing.
_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
_JPEG = b"\xff\xd8\xff" + b"pixels"


def tqdm_stub():
    """A progress bar that only has to accept ``update``."""

    class _Stub:
        def update(self, _n: int = 1) -> None:
            pass

    return _Stub()


# lmarena-ai/VisionArena-Chat: every turn is wrapped in a single-element list,
# images carry inline bytes, and the responses come from an arena model.
_VISIONARENA_ROW: dict[str, Any] = {
    "conversation_id": "cbda71fbc23545faaa3238e6f4177ef6",
    "model": "gpt-4o-mini-2024-07-18",
    "num_turns": 2,
    "language": "English",
    "images": [{"bytes": _PNG, "path": "970cecefeccddc5e1a97e354952c42f4.png"}],
    "conversation": [
        [{"content": "What's this?", "role": "user"}],
        [{"content": "<arena response to drop>", "role": "assistant"}],
        [{"content": "It is seaweed, not a snake.", "role": "user"}],
        [{"content": "<arena response to drop>", "role": "assistant"}],
    ],
}


# ---------------------------------------------------------------------------
# export_visionarena.py: conversation shape
# ---------------------------------------------------------------------------


def test_flatten_conversation_unwraps_the_single_element_lists():
    assert export.flatten_conversation(_VISIONARENA_ROW["conversation"]) == [
        {"content": "What's this?", "role": "user"},
        {"content": "<arena response to drop>", "role": "assistant"},
        {"content": "It is seaweed, not a snake.", "role": "user"},
        {"content": "<arena response to drop>", "role": "assistant"},
    ]


def test_flatten_conversation_tolerates_already_flat_and_junk_entries():
    conversation = [{"role": "user", "content": "hi"}, None, ["nested-string"], 7]
    assert export.flatten_conversation(conversation) == [
        {"role": "user", "content": "hi"}
    ]
    assert export.flatten_conversation(None) == []
    assert export.flatten_conversation("not a list") == []


def test_extract_prompt_turns_drops_the_arena_responses():
    turns = export.flatten_conversation(_VISIONARENA_ROW["conversation"])
    assert export.extract_prompt_turns(turns) == [
        {"role": "user", "content": "What's this?"},
        {"role": "user", "content": "It is seaweed, not a snake."},
    ]


def test_extract_prompt_turns_keeps_system_turns_and_the_from_value_schema():
    turns = [
        {"from": "system", "value": "You are terse."},
        {"from": "human", "value": "Hi"},
        {"from": "gpt", "value": "<drop>"},
    ]
    assert export.extract_prompt_turns(turns) == [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "Hi"},
    ]


def test_extract_prompt_turns_caps_user_turns():
    turns = export.flatten_conversation(_VISIONARENA_ROW["conversation"])
    assert export.extract_prompt_turns(turns, max_turns=1) == [
        {"role": "user", "content": "What's this?"}
    ]


def test_extract_prompt_turns_rejects_a_conversation_with_an_empty_user_turn():
    # Keeping the empty turn would give the model nothing to answer and shift
    # every later turn onto the wrong image, so the row is dropped entirely.
    turns = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "<drop>"},
        {"role": "user", "content": ""},
    ]
    assert export.extract_prompt_turns(turns) == []


def test_extract_prompt_turns_drops_a_trailing_system_turn():
    turns = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "<drop>"},
        {"role": "system", "content": "Be terse."},
    ]
    assert export.extract_prompt_turns(turns) == [{"role": "user", "content": "Hi"}]


def test_extract_prompt_turns_needs_at_least_one_user_turn():
    assert export.extract_prompt_turns([{"role": "system", "content": "Hi"}]) == []
    assert export.extract_prompt_turns([]) == []


# ---------------------------------------------------------------------------
# export_visionarena.py: image materialization
# ---------------------------------------------------------------------------


def test_write_images_materializes_inline_bytes(tmp_path):
    paths = export.write_images(_VISIONARENA_ROW["images"], tmp_path)
    assert [p.name for p in paths] == ["970cecefeccddc5e1a97e354952c42f4.png"]
    assert paths[0].read_bytes() == _PNG


def test_write_images_leaves_an_existing_file_alone(tmp_path):
    # Names are content hashes, so a rerun (or an image shared between
    # conversations) must not rewrite what is already on disk.
    existing = tmp_path / "img.png"
    existing.write_bytes(b"already here")
    paths = export.write_images([{"bytes": _PNG, "path": "img.png"}], tmp_path)
    assert paths == [existing]
    assert existing.read_bytes() == b"already here"


def test_write_images_cannot_escape_the_image_dir(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    paths = export.write_images(
        [{"bytes": _PNG, "path": "../../etc/evil.png"}], image_dir
    )
    assert paths == [image_dir / "evil.png"]
    assert not (tmp_path.parent / "etc").exists()


def test_write_images_skips_entries_with_no_bytes(tmp_path):
    images = [
        {"bytes": None, "path": "missing.png"},
        {"path": "also-missing.png"},
        "not a dict",
        {"bytes": _JPEG, "path": "real.jpg"},
    ]
    assert [p.name for p in export.write_images(images, tmp_path)] == ["real.jpg"]


def test_write_images_names_an_unnamed_image_by_content_hash(tmp_path):
    paths = export.write_images([{"bytes": _JPEG}], tmp_path)
    assert len(paths) == 1
    assert paths[0].suffix == ".jpg"
    # Same bytes must land on the same name, so the write stays idempotent.
    assert export.write_images([{"bytes": _JPEG}], tmp_path) == paths


@pytest.mark.parametrize(
    ("data", "suffix"),
    [
        (_PNG, ".png"),
        (_JPEG, ".jpg"),
        (b"GIF89a...", ".gif"),
        (b"RIFF1234WEBPmore", ".webp"),
        (b"totally unknown", ".img"),
    ],
)
def test_guess_suffix(data, suffix):
    assert export._guess_suffix(data) == suffix


# ---------------------------------------------------------------------------
# export_visionarena.py: image attachment and whole-row conversion
# ---------------------------------------------------------------------------


def test_attach_images_targets_only_the_first_user_turn(tmp_path):
    image = tmp_path / "a.png"
    turns = [
        {"role": "user", "content": "First"},
        {"role": "user", "content": "Second"},
    ]
    attached = export.attach_images(turns, [image])

    assert attached[0]["content"] == [
        {"type": "image", "path": str(image.resolve())},
        {"type": "text", "text": "First"},
    ]
    # A later turn keeps plain-string content; the image is already in context.
    assert attached[1]["content"] == "Second"


def test_attach_images_is_a_no_op_without_images():
    turns = [{"role": "user", "content": "Text only"}]
    assert export.attach_images(turns, []) == turns


def test_build_row_converts_a_real_visionarena_row(tmp_path):
    row = export.build_row(
        _VISIONARENA_ROW, tmp_path, max_turns=None, require_images=False
    )
    assert row is not None
    assert row["conversation_id"] == _VISIONARENA_ROW["conversation_id"]

    conversations = row["conversations"]
    assert [turn["role"] for turn in conversations] == ["user", "user"]
    image_part, text_part = conversations[0]["content"]
    assert image_part["type"] == "image"
    assert Path(image_part["path"]).read_bytes() == _PNG
    assert text_part == {"type": "text", "text": "What's this?"}


def test_build_row_can_require_images(tmp_path):
    text_only = {**_VISIONARENA_ROW, "images": []}
    assert (
        export.build_row(text_only, tmp_path, max_turns=None, require_images=True)
        is None
    )
    kept = export.build_row(text_only, tmp_path, max_turns=None, require_images=False)
    assert kept is not None
    assert kept["conversations"][0]["content"] == "What's this?"


def test_load_exported_ids_ignores_malformed_lines(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"conversation_id": "a"}\nnot json\n{"conversations": []}\n'
        '{"conversation_id": "b"}\n',
        encoding="utf-8",
    )
    assert export.load_exported_ids(path) == {"a", "b"}
    assert export.load_exported_ids(tmp_path / "absent.jsonl") == set()


# ---------------------------------------------------------------------------
# export_visionarena.py: reading an already-downloaded copy
# ---------------------------------------------------------------------------


def _write_local_copy(directory: Path, *, decode: bool) -> None:
    """Write a one-row parquet copy with VisionArena's schema.

    ``decode`` picks how the images column is declared, which is what decides
    whether rows arrive as ``{bytes, path}`` dicts or as PIL objects.
    """
    features = Features(
        {
            "images": [Image(decode=decode)],
            "conversation_id": Value("string"),
            "conversation": [[{"content": Value("string"), "role": Value("string")}]],
            "language": Value("string"),
        }
    )

    # A real PNG: a decode=True column has to be decodable by PIL to load.
    def _chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(b"\x00\x01\x02\x03"))
        + _chunk(b"IEND", b"")
    )
    directory.mkdir(parents=True, exist_ok=True)
    Dataset.from_dict(
        {
            "images": [[{"bytes": png, "path": "abc.png"}]],
            "conversation_id": ["c0"],
            "conversation": [[[{"content": "What's this?", "role": "user"}]]],
            "language": ["English"],
        },
        features=features,
    ).to_parquet(directory / "train-00000-of-00001.parquet")


@pytest.mark.parametrize("decode", [False, True])
def test_load_local_dataset_always_yields_writable_image_bytes(tmp_path, decode):
    """Images must survive either way the column can be declared.

    HF defaults Image columns to ``decode=True``, which yields PIL objects that
    ``write_images`` has no bytes to write and skips silently -- an export of
    prompts with every image missing. The loader pins ``decode=False`` so a
    re-encoded local copy cannot cause that.
    """
    _write_local_copy(tmp_path / "copy", decode=decode)

    row = next(iter(export.load_local_dataset(tmp_path / "copy")))
    assert isinstance(row["images"][0], dict)

    image_dir = tmp_path / "images"
    image_dir.mkdir()
    written = export.write_images(row["images"], image_dir)
    assert len(written) == 1
    assert written[0].is_file()


def test_load_local_dataset_finds_shards_in_nested_directories(tmp_path):
    """The HF cache nests shards under the snapshot, e.g. ``snapshots/<rev>/data``."""
    _write_local_copy(tmp_path / "snapshots" / "deadbeef" / "data", decode=False)

    assert next(iter(export.load_local_dataset(tmp_path)))["conversation_id"] == "c0"


def test_load_local_dataset_rejects_a_directory_with_no_data(tmp_path):
    """A cache entry holding only README.md must not look like a usable copy."""
    (tmp_path / "README.md").write_text("card only", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No dataset found"):
        export.load_local_dataset(tmp_path)


def test_resolve_dataset_dir_rejects_a_missing_explicit_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        export.resolve_dataset_dir(tmp_path / "absent")


def test_resolve_dataset_dir_uses_an_explicit_path_verbatim(tmp_path):
    assert export.resolve_dataset_dir(tmp_path) == tmp_path


# ---------------------------------------------------------------------------
# export_nemotron_vlm.py: row shape
# ---------------------------------------------------------------------------

# nvidia/Llama-Nemotron-VLM-Dataset-v1: one image per row, a single human/gpt
# exchange, and an <image> placeholder marking the image's place in the prompt.
_NEMOTRON_ROW: dict[str, Any] = {
    "id": "22935085-3b90-4348-a4b1-6985ec7da67e",
    "image": "502450.png",
    "conversations": [
        {"from": "human", "value": "<image>\nExtract all visible text."},
        {"from": "gpt", "value": "<dataset response to drop>"},
    ],
}


def test_build_row_converts_a_real_nemotron_row(tmp_path):
    row = nemotron.build_row(_NEMOTRON_ROW, "ocr_1", tmp_path / "502450.png")

    assert row is not None
    assert row["conversation_id"] == "ocr_1/22935085-3b90-4348-a4b1-6985ec7da67e"
    # The gpt turn is off-policy and must not survive into the export.
    assert [turn["role"] for turn in row["conversations"]] == ["user"]
    assert row["conversations"][0]["content"] == [
        {"type": "image", "path": str(tmp_path / "502450.png")},
        {"type": "text", "text": "Extract all visible text."},
    ]


def test_prompt_parts_keeps_the_placeholder_position(tmp_path):
    """A mid-prompt placeholder must not be hoisted to the front."""
    parts = nemotron.prompt_parts("Compare <image> with the table.", tmp_path / "i.png")

    assert [part["type"] for part in parts] == ["text", "image", "text"]
    assert parts[0]["text"] == "Compare"
    assert parts[2]["text"] == "with the table."


def test_build_row_rejects_a_prompt_that_is_only_an_image():
    row = dict(_NEMOTRON_ROW)
    row["conversations"] = [{"from": "human", "value": "<image>"}]

    assert nemotron.build_row(row, "ocr_1", Path("/x/i.png")) is None


def test_build_row_rejects_a_row_with_no_user_turn():
    row = dict(_NEMOTRON_ROW)
    row["conversations"] = [{"from": "gpt", "value": "answer"}]

    assert nemotron.build_row(row, "ocr_1", Path("/x/i.png")) is None


def test_nemotron_export_is_accepted_by_prepare_data(tmp_path):
    """The handoff: what this export writes, prepare_data.py has to accept."""
    row = nemotron.build_row(_NEMOTRON_ROW, "ocr_1", tmp_path / "502450.png")
    assert row is not None
    # Regeneration appends the on-policy answer; that is what gets prepared.
    conversations = [*row["conversations"], {"role": "assistant", "content": "text"}]

    adapted = _adapt_conv_for_vllm(conversations)

    assert [part["type"] for part in adapted[0]["content"]] == ["image_url", "text"]
    assert adapted[0]["content"][0]["image_url"]["url"].startswith("file://")


def test_regeneration_can_read_the_nemotron_export(tmp_path):
    row = nemotron.build_row(_NEMOTRON_ROW, "ocr_1", tmp_path / "502450.png")
    assert row is not None

    turns = regen.prompt_turns(row)

    assert [turn["role"] for turn in turns] == ["user"]
    assert turns[0]["content"][0]["type"] == "image"


# ---------------------------------------------------------------------------
# export_nemotron_vlm.py: fraction sampling
# ---------------------------------------------------------------------------


def _kept(fraction: float, ids: list[str], partition: str = "ocr_1") -> set[str]:
    return {i for i in ids if nemotron.keeps_row(partition, i, fraction, seed=0)}


@pytest.fixture
def row_ids() -> list[str]:
    return [f"{i:08d}-uuid" for i in range(20_000)]


@pytest.mark.parametrize("fraction", [0.01, 0.1, 0.5])
def test_keeps_row_selects_about_the_requested_fraction(row_ids, fraction):
    kept = _kept(fraction, row_ids)

    # Binomial noise at n=20k is well inside 15% relative at these fractions.
    assert kept
    assert abs(len(kept) / len(row_ids) - fraction) < 0.15 * fraction


def test_keeps_row_is_nested_so_raising_the_fraction_only_adds(row_ids):
    """The property that makes topping up cheap: no row already exported is lost.

    A shuffle-and-slice sample would pick a different subset when the fraction
    changes, discarding the images the previous run extracted.
    """
    tenth = _kept(0.1, row_ids)
    fifth = _kept(0.2, row_ids)

    assert tenth < fifth


def test_keeps_row_is_deterministic(row_ids):
    assert _kept(0.1, row_ids) == _kept(0.1, row_ids)


def test_keeps_row_differs_between_partitions(row_ids):
    """Partitions must not share a selection pattern."""
    assert _kept(0.1, row_ids, "ocr_1") != _kept(0.1, row_ids, "ocr_4")


def test_keeps_row_keeps_everything_at_fraction_one(row_ids):
    assert _kept(1.0, row_ids) == set(row_ids)


# ---------------------------------------------------------------------------
# export_nemotron_vlm.py: image naming and partition selection
# ---------------------------------------------------------------------------


def test_destination_name_flattens_nested_member_names():
    name = nemotron.destination_name("data/train/project-26/0000160/99833.md.jpg")

    assert name == "data__train__project-26__0000160__99833.md.jpg"
    assert Path(name).name == name


@pytest.mark.parametrize(
    "member", ["../../etc/passwd", "/etc/passwd", "..", "a/../../b", ""]
)
def test_destination_name_cannot_escape_the_image_dir(member):
    """Member names come from the archive, so they must not choose the path."""
    name = nemotron.destination_name(member)

    assert Path(name).name == name
    assert ".." not in Path(name).parts
    assert (Path("/images") / name).resolve().parent == Path("/images")


def test_destination_name_shortens_an_overlong_name():
    name = nemotron.destination_name("x/" * 400 + "img.png")

    assert len(name) <= 200
    assert Path(name).name == name


def _make_partition(
    directory: Path, name: str, rows: list[dict], *, with_images: bool
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    if with_images:
        image_dir = directory / f"{name}_images"
        image_dir.mkdir(exist_ok=True)
        with tarfile.open(image_dir / "shard_000001.tar", "w") as archive:
            for row in rows:
                info = tarfile.TarInfo(row["image"])
                info.size = len(_PNG)
                archive.addfile(info, io.BytesIO(_PNG))


@pytest.fixture
def nemotron_dir(tmp_path) -> Path:
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.png") for i in range(4)]
    _make_partition(tmp_path, "ocr_1", rows, with_images=True)
    _make_partition(tmp_path, "vqa_1", rows, with_images=False)
    return tmp_path


def test_select_partitions_defaults_to_those_with_images(nemotron_dir):
    chosen = nemotron.select_partitions(nemotron_dir, None)

    assert [part["name"] for part in chosen] == ["ocr_1"]


def test_select_partitions_rejects_a_partition_without_images(nemotron_dir):
    """Exporting one would emit prompts pointing at files that do not exist."""
    with pytest.raises(ValueError, match="have no images"):
        nemotron.select_partitions(nemotron_dir, "ocr_1,vqa_1")


def test_select_partitions_rejects_an_unknown_partition(nemotron_dir):
    with pytest.raises(ValueError, match="Unknown partition"):
        nemotron.select_partitions(nemotron_dir, "ocr_99")


def test_select_partitions_keeps_the_requested_order(nemotron_dir):
    rows = [dict(_NEMOTRON_ROW, id="a", image="a.png")]
    _make_partition(nemotron_dir, "ocr_4", rows, with_images=True)

    chosen = nemotron.select_partitions(nemotron_dir, "ocr_4,ocr_1")

    assert [part["name"] for part in chosen] == ["ocr_4", "ocr_1"]


def test_count_rows_prefers_the_sidecar_index(tmp_path):
    """The dataset ships uint64 offsets per row plus a terminator."""
    partition = tmp_path / "ocr_1.jsonl"
    partition.write_text('{"a": 1}\n', encoding="utf-8")
    # Claims 7 rows, which only the index could report.
    partition.with_suffix(".jsonl.idx").write_bytes(b"\x00" * 8 * 8)

    assert nemotron.count_rows(partition) == 7


def test_count_rows_falls_back_to_counting_lines(tmp_path):
    partition = tmp_path / "ocr_1.jsonl"
    partition.write_text('{"a": 1}\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")

    assert nemotron.count_rows(partition) == 3


def test_export_partition_extracts_only_the_selected_images(nemotron_dir, tmp_path):
    """End to end over a real TAR: selection, extraction and row writing."""
    partition = nemotron.select_partitions(nemotron_dir, "ocr_1")[0]
    wanted = nemotron.select_rows(partition, fraction=1.0, seed=0, exported_ids={"x"})
    # Keep two of the four images to prove the others are left in the archive.
    wanted = dict(list(wanted.items())[:2])

    outfile = tmp_path / "out" / "prompts.jsonl"
    outfile.parent.mkdir()
    image_root = tmp_path / "images"
    with outfile.open("w", encoding="utf-8") as handle:
        written, skipped, missing = nemotron.export_partition(
            partition, wanted, image_root, handle, tqdm_stub()
        )

    assert (written, skipped, missing) == (2, 0, 0)
    assert sorted(p.name for p in (image_root / "ocr_1").iterdir()) == sorted(wanted)
    rows = [json.loads(line) for line in outfile.read_text().splitlines()]
    assert len(rows) == 2
    for row in rows:
        image = row["conversations"][0]["content"][0]
        assert Path(image["path"]).is_file()


def test_export_partition_reports_images_missing_from_the_archive(
    nemotron_dir, tmp_path
):
    """An incomplete download must be counted, not silently exported."""
    partition = nemotron.select_partitions(nemotron_dir, "ocr_1")[0]
    wanted = {"absent.png": [dict(_NEMOTRON_ROW, image="absent.png")]}

    outfile = tmp_path / "prompts.jsonl"
    with outfile.open("w", encoding="utf-8") as handle:
        written, _, missing = nemotron.export_partition(
            partition, wanted, tmp_path / "images", handle, tqdm_stub()
        )

    assert (written, missing) == (0, 1)
    assert outfile.read_text() == ""


def test_select_rows_skips_ids_already_exported(nemotron_dir):
    partition = nemotron.select_partitions(nemotron_dir, "ocr_1")[0]

    all_rows = nemotron.select_rows(partition, 1.0, 0, set())
    minus_one = nemotron.select_rows(partition, 1.0, 0, {"ocr_1/id0"})

    assert sum(map(len, all_rows.values())) == 4
    assert sum(map(len, minus_one.values())) == 3


# ---------------------------------------------------------------------------
# regenerate_vlm_responses.py: input handling
# ---------------------------------------------------------------------------


def _exported_row(tmp_path) -> dict:
    return export.build_row(
        _VISIONARENA_ROW, tmp_path, max_turns=None, require_images=False
    )


def test_prompt_turns_reads_the_exported_row(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    assert [turn["role"] for turn in turns] == ["user", "user"]
    assert turns[0]["content"][0]["type"] == "image"


def test_prompt_turns_drops_assistant_turns_from_an_already_regenerated_row():
    row = {
        "conversations": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<not from the target model>"},
            {"role": "user", "content": "Again"},
        ]
    }
    assert regen.prompt_turns(row) == [
        {"role": "user", "content": "Hi"},
        {"role": "user", "content": "Again"},
    ]


def test_missing_image_paths_reports_only_absent_files(tmp_path):
    present = tmp_path / "here.png"
    present.write_bytes(_PNG)
    turns = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(present)},
                {"type": "image", "path": str(tmp_path / "gone.png")},
                {"type": "text", "text": "hi"},
            ],
        },
        {"role": "user", "content": "plain string"},
    ]
    assert regen.missing_image_paths(turns) == [str(tmp_path / "gone.png")]


def test_assistant_turn_preserves_reasoning_under_both_keys():
    turn = regen.assistant_turn({"content": "answer", "reasoning_content": "because"})
    # prepare_data.py's normalizer reads either key.
    assert turn == {
        "role": "assistant",
        "content": "answer",
        "thinking": "because",
        "reasoning_content": "because",
    }


def test_build_work_items_skips_completed_and_missing_images(tmp_path):
    good = _exported_row(tmp_path)
    done = {**good, "conversation_id": "already-done"}
    broken = {
        "conversation_id": "broken",
        "conversations": [
            {
                "role": "user",
                "content": [{"type": "image", "path": str(tmp_path / "nope.png")}],
            }
        ],
    }
    items = regen.build_work_items([good, done, broken], {"already-done"})
    assert [item["conversation_id"] for item in items] == [
        _VISIONARENA_ROW["conversation_id"]
    ]


def test_iter_input_rows_honors_limit_and_skips_bad_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\n\nbroken\n{"a": 2}\n{"a": 3}\n', encoding="utf-8")
    assert regen.iter_input_rows(path, None) == [{"a": 1}, {"a": 2}, {"a": 3}]
    assert regen.iter_input_rows(path, 2) == [{"a": 1}, {"a": 2}]


@pytest.mark.parametrize(
    ("num_completed", "expected"),
    [(0, 2), (1, 1), (2, 0)],
)
def test_count_remaining_prints_the_count_without_an_endpoint(
    tmp_path, monkeypatch, capsys, num_completed, expected
):
    """--count-remaining must answer from disk alone.

    The pipeline calls this to decide whether starting a server is worth it, so
    reaching for the endpoint here would defeat the purpose. detect_model is
    replaced with a raiser to prove it is never consulted.
    """
    rows = [{**_exported_row(tmp_path), "conversation_id": f"c{i}"} for i in range(2)]
    data = tmp_path / "prompts.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    outfile = tmp_path / "conversations.jsonl"
    outfile.write_text(
        "".join(json.dumps(rows[i]) + "\n" for i in range(num_completed)),
        encoding="utf-8",
    )

    async def _no_endpoint(_endpoint):
        raise AssertionError("--count-remaining must not contact the endpoint")

    monkeypatch.setattr(regen, "detect_model", _no_endpoint)
    monkeypatch.setattr(
        "sys.argv",
        [
            "regenerate_vlm_responses.py",
            "--data",
            str(data),
            "--outfile",
            str(outfile),
            "--resume",
            "--count-remaining",
        ],
    )

    asyncio.run(regen.main())

    # stdout carries the bare integer so a shell can capture it directly.
    assert capsys.readouterr().out.strip() == str(expected)


# ---------------------------------------------------------------------------
# regenerate_vlm_responses.py: the regeneration loop
# ---------------------------------------------------------------------------


class _FakeEndpoint:
    """Records the payloads it is sent and replies with canned generations."""

    def __init__(self, replies: list[dict]):
        self.replies = replies
        self.payloads: list[dict] = []

    async def __call__(self, payload: dict) -> dict:
        # Copy: the caller keeps mutating the same `messages` list as it goes.
        self.payloads.append(json.loads(json.dumps(payload)))
        return self.replies[len(self.payloads) - 1]


def _reply(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {"message": {"content": content}, "finish_reason": finish_reason},
        ]
    }


def _run_regeneration(post_fn, turns):
    return asyncio.run(
        regen.regenerate_conversation(
            post_fn, turns, model="target", max_tokens=64, sampling_params={}
        )
    )


def test_regeneration_sends_image_url_but_returns_path_parts(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("A snake."), _reply("Seaweed, then.")])

    conversations, truncated = _run_regeneration(endpoint, turns)
    assert not truncated

    # On the wire the image is an image_url part pointing at a file:// URL.
    sent_content = endpoint.payloads[0]["messages"][0]["content"]
    assert sent_content[0]["type"] == "image_url"
    assert sent_content[0]["image_url"]["url"].startswith("file:///")

    # In the output it keeps the `path` form, which is what prepare_data.py takes.
    assert conversations[0]["content"][0]["type"] == "image"
    assert Path(conversations[0]["content"][0]["path"]).is_file()

    assert [turn["role"] for turn in conversations] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert conversations[1]["content"] == "A snake."
    assert conversations[3]["content"] == "Seaweed, then."


def test_regeneration_conditions_each_turn_on_the_models_own_response(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("first"), _reply("second")])

    _run_regeneration(endpoint, turns)

    # The second request must carry the model's own first response, not the
    # arena's -- that is what keeps the assistant history on-policy.
    second_request = endpoint.payloads[1]["messages"]
    assert [m["role"] for m in second_request] == ["user", "assistant", "user"]
    assert second_request[1]["content"] == "first"
    assert "<arena response to drop>" not in json.dumps(endpoint.payloads)


def test_regeneration_stops_after_a_truncated_response(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("cut off", finish_reason="length")])

    conversations, truncated = _run_regeneration(endpoint, turns)

    assert truncated
    # Only one request: conditioning a later turn on a cut-off response would
    # not reflect anything the model would actually do at serving time.
    assert len(endpoint.payloads) == 1
    assert [turn["role"] for turn in conversations] == ["user", "assistant"]


def test_regeneration_rejects_an_empty_generation(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("")])

    with pytest.raises(ValueError, match="empty assistant generation"):
        _run_regeneration(endpoint, turns)


def test_regeneration_does_not_let_sampling_params_override_owned_keys(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("ok"), _reply("ok")])

    asyncio.run(
        regen.regenerate_conversation(
            endpoint,
            turns,
            model="target",
            max_tokens=64,
            sampling_params={"model": "hijacked", "max_tokens": 1, "temperature": 0.7},
        )
    )
    payload = endpoint.payloads[0]
    assert payload["model"] == "target"
    assert payload["max_tokens"] == 64
    assert payload["temperature"] == 0.7


# ---------------------------------------------------------------------------
# regenerate_vlm_responses.py: the worker, driven over a fake session
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: dict, ok: bool = True, status: int = 200):
        self.payload = payload
        self.ok = ok
        self.status = status

    async def json(self) -> dict:
        return self.payload

    async def text(self) -> str:
        return json.dumps(self.payload)


class _FakeSession:
    """Mimics the ``async with session.post(...) as response`` shape."""

    def __init__(self, responses: list[_FakeResponse]):
        self.responses = list(responses)

    # Keyword name is fixed by aiohttp's API, so it shadows the json module here.
    def post(self, endpoint: str, json: dict | None = None):
        response = self.responses.pop(0)

        class _Ctx:
            async def __aenter__(self):
                return response

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


class _Args:
    endpoint = "http://fake/v1/chat/completions"
    max_retries = 0
    model = "target"
    max_tokens = 64
    sampling_params: dict = {}


class _NullProgress:
    def update(self, _n=1):
        pass

    def set_postfix(self, *_args, **_kwargs):
        pass


def _run_worker(responses: list[_FakeResponse], turns: list[dict], tmp_path: Path):
    out_path = tmp_path / "out.jsonl"
    err_path = tmp_path / "out.errors.jsonl"
    stats = {
        "ok": 0,
        "errors": 0,
        "truncated": 0,
        "requests": 0,
        "total_request_s": 0.0,
        "start_time": 0.0,
    }

    async def scenario(out_fh, err_fh):
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put({"conversation_id": "conv-1", "turns": turns})
        await queue.put(None)
        await regen.worker(
            _FakeSession(responses),
            queue,
            _Args(),
            out_fh,
            err_fh,
            _NullProgress(),
            stats,
        )

    with (
        out_path.open("w", encoding="utf-8") as out_fh,
        err_path.open("w", encoding="utf-8") as err_fh,
    ):
        asyncio.run(scenario(out_fh, err_fh))
    return stats, out_path, err_path


def test_worker_writes_a_regenerated_row(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    stats, out_path, err_path = _run_worker(
        [_FakeResponse(_reply("A snake.")), _FakeResponse(_reply("Seaweed."))],
        turns,
        tmp_path,
    )

    assert stats == {
        "ok": 1,
        "errors": 0,
        "truncated": 0,
        "requests": 2,
        "total_request_s": stats["total_request_s"],
        "start_time": 0.0,
    }
    assert err_path.read_text() == ""

    row = json.loads(out_path.read_text())
    assert row["conversation_id"] == "conv-1"
    assert row["metadata"] == {"model": "target", "truncated": False}
    assert [t["role"] for t in row["conversations"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    # The written row is what --resume keys on.
    assert regen.load_completed_ids(out_path) == {"conv-1"}


def test_worker_sends_failures_to_the_error_file(tmp_path):
    turns = regen.prompt_turns(_exported_row(tmp_path))
    stats, out_path, err_path = _run_worker(
        [_FakeResponse({"detail": "bad request"}, ok=False, status=400)],
        turns,
        tmp_path,
    )

    assert stats["ok"] == 0
    assert stats["errors"] == 1
    # Nothing half-written: a failed conversation must not enter the training input.
    assert out_path.read_text() == ""
    assert regen.load_completed_ids(out_path) == set()

    error = json.loads(err_path.read_text())
    assert error["conversation_id"] == "conv-1"
    assert "400" in error["error"]


# ---------------------------------------------------------------------------
# The handoff to prepare_data.py
# ---------------------------------------------------------------------------


def test_regenerated_conversations_are_accepted_by_prepare_data(tmp_path):
    """The output must survive prepare_data.py's own content-part adapter."""
    turns = regen.prompt_turns(_exported_row(tmp_path))
    endpoint = _FakeEndpoint([_reply("A snake."), _reply("Seaweed, then.")])
    conversations, _ = _run_regeneration(endpoint, turns)

    adapted = _adapt_conv_for_vllm(conversations)
    assert adapted[0]["content"][0]["type"] == "image_url"
    assert adapted[0]["content"][0]["image_url"]["url"].startswith("file:///")


def test_prepare_data_would_reject_the_wire_form(tmp_path):
    """Why the script keeps two views instead of writing what it sent.

    ``prepare_data.py`` handles ``{"type": "image", "path": ...}``, not the
    ``image_url`` form vLLM is sent, so writing the wire form would fail the
    whole preprocessing run.
    """
    wire = _adapt_conv_for_vllm(
        [
            {
                "role": "user",
                "content": [{"type": "image", "path": str(tmp_path / "a.png")}],
            }
        ]
    )
    with pytest.raises(NotImplementedError, match="Unknown content part"):
        _adapt_conv_for_vllm(wire)


# ---------------------------------------------------------------------------
# download_nemotron_images.py + the loose-image export path
#
# The partitions whose images the repo does not ship (vqa_1 and friends) get
# them from OpenImages instead, as plain files rather than TAR shards. The
# assertions that matter here are the agreement ones: the downloader has to
# choose exactly the rows the export will later ask for, or a run downloads one
# subset and exports another.
# ---------------------------------------------------------------------------


def _make_loose_partition(
    directory: Path, name: str, rows: list[dict], present: list[str]
) -> None:
    """A partition whose images are plain files, only some of them present."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    image_dir = directory / f"{name}_images"
    image_dir.mkdir(exist_ok=True)
    for image in present:
        (image_dir / image).write_bytes(_JPEG)


def test_the_two_scripts_agree_on_which_partitions_are_downloadable():
    """The export names them only to suggest the downloader; a drift misleads."""
    assert set(download.IMAGE_SOURCES) == nemotron._DOWNLOADABLE


def test_download_and_export_select_the_same_rows(tmp_path):
    """The point of sharing keeps_row: one subset, not two."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(400)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=[])

    to_download = download.wanted_images(tmp_path, "vqa_1", fraction=0.25, seed=0)

    # What the export would ask for, over the same rows and fraction.
    partition = {"name": "vqa_1", "path": tmp_path / "vqa_1.jsonl"}
    to_export = set(nemotron.select_rows(partition, 0.25, 0, set()))

    assert to_download == to_export
    assert 0 < len(to_download) < len(rows)


def test_download_selection_is_nested_so_raising_the_fraction_only_adds(tmp_path):
    """Why a top-up refetches nothing: the smaller selection is a subset."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(400)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=[])

    small = download.wanted_images(tmp_path, "vqa_1", fraction=0.1, seed=0)
    large = download.wanted_images(tmp_path, "vqa_1", fraction=0.3, seed=0)

    assert small < large


def test_download_deduplicates_images_shared_by_several_rows(tmp_path):
    """captioning_2 asks several questions about one image; fetch it once."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image="same.jpg") for i in range(20)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=[])

    assert download.wanted_images(tmp_path, "vqa_1", 1.0, 0) == {"same.jpg"}


def test_already_present_is_what_makes_a_download_resumable(tmp_path):
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(4)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=["0.jpg", "2.jpg"])

    names = download.wanted_images(tmp_path, "vqa_1", 1.0, 0)
    present = download.already_present(tmp_path / "vqa_1_images", names)

    assert present == {"0.jpg", "2.jpg"}
    assert names - present == {"1.jpg", "3.jpg"}


def test_downloader_rejects_a_partition_with_no_derivable_urls(tmp_path):
    """ChartQA and friends have to be fetched by hand; say so rather than 404."""
    _make_loose_partition(tmp_path, "vqa_4", [], present=[])

    with pytest.raises(ValueError, match="do not have derivable image URLs"):
        download.select_partitions(tmp_path, "vqa_4")


def test_downloader_rejects_a_partition_with_no_local_metadata(tmp_path):
    with pytest.raises(FileNotFoundError, match="No JSONL for partition"):
        download.select_partitions(tmp_path, "vqa_1")


def test_loose_images_make_a_shipped_imageless_partition_exportable(tmp_path):
    """The whole reason for the download: vqa_1 becomes usable."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(3)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=["0.jpg", "1.jpg", "2.jpg"])

    chosen = nemotron.select_partitions(tmp_path, "vqa_1")

    assert len(chosen) == 1
    assert chosen[0]["has_images"] is True
    assert chosen[0]["shards"] == []
    assert chosen[0]["loose_dir"] == tmp_path / "vqa_1_images"


def test_export_references_downloaded_images_in_place(tmp_path):
    """Copying them would be a second copy of a selection that reaches 378 GB."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(2)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=["0.jpg", "1.jpg"])
    partition = nemotron.select_partitions(tmp_path, "vqa_1")[0]
    wanted = nemotron.select_rows(partition, 1.0, 0, set())

    image_root = tmp_path / "export_images"
    outfile = tmp_path / "prompts.jsonl"
    with outfile.open("w", encoding="utf-8") as handle:
        written, skipped, missing = nemotron.export_partition(
            partition, wanted, image_root, handle, tqdm_stub()
        )

    assert (written, skipped, missing) == (2, 0, 0)
    assert not image_root.exists()
    for line in outfile.read_text().splitlines():
        image = json.loads(line)["conversations"][0]["content"][0]
        assert Path(image["path"]).parent == tmp_path / "vqa_1_images"
        assert Path(image["path"]).is_file()


def test_export_counts_rows_whose_download_404ed(tmp_path):
    """OpenImages has removed keys over time; those rows must not be emitted."""
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.jpg") for i in range(3)]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=["0.jpg", "2.jpg"])
    partition = nemotron.select_partitions(tmp_path, "vqa_1")[0]
    wanted = nemotron.select_rows(partition, 1.0, 0, set())

    outfile = tmp_path / "prompts.jsonl"
    with outfile.open("w", encoding="utf-8") as handle:
        written, _, missing = nemotron.export_partition(
            partition, wanted, tmp_path / "images", handle, tqdm_stub()
        )

    assert (written, missing) == (2, 1)


def test_a_loose_image_name_cannot_escape_its_directory(tmp_path):
    """The image field comes from the dataset, so it is confined before use."""
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(_JPEG)
    rows = [dict(_NEMOTRON_ROW, id="id0", image="../secret.jpg")]
    _make_loose_partition(tmp_path, "vqa_1", rows, present=["decoy.jpg"])
    partition = nemotron.select_partitions(tmp_path, "vqa_1")[0]
    wanted = nemotron.select_rows(partition, 1.0, 0, set())

    outfile = tmp_path / "prompts.jsonl"
    with outfile.open("w", encoding="utf-8") as handle:
        written, _, missing = nemotron.export_partition(
            partition, wanted, tmp_path / "images", handle, tqdm_stub()
        )

    assert (written, missing) == (0, 1)
    assert outfile.read_text() == ""


def test_image_source_adds_a_location_without_hiding_the_shipped_tars(tmp_path):
    """Treating it as an override would lose the tar-shipped partitions."""
    dataset_dir = tmp_path / "dataset"
    rows = [dict(_NEMOTRON_ROW, id=f"id{i}", image=f"{i}.png") for i in range(2)]
    _make_partition(dataset_dir, "ocr_1", rows, with_images=True)
    _make_partition(dataset_dir, "vqa_1", rows, with_images=False)

    # vqa_1's images were downloaded somewhere else entirely.
    elsewhere = tmp_path / "downloaded"
    elsewhere.mkdir()
    (elsewhere / "vqa_1_images").mkdir()
    for row in rows:
        (elsewhere / "vqa_1_images" / row["image"]).write_bytes(_JPEG)

    found = {
        part["name"]: part
        for part in nemotron.discover_partitions(dataset_dir, elsewhere)
    }

    assert found["ocr_1"]["has_images"] is True
    assert found["ocr_1"]["shards"], "the shipped tars must still be found"
    assert found["vqa_1"]["has_images"] is True
    assert found["vqa_1"]["loose_dir"] == elsewhere / "vqa_1_images"


def test_a_directory_of_only_tars_is_not_mistaken_for_loose_images(tmp_path):
    rows = [dict(_NEMOTRON_ROW, id="id0", image="0.png")]
    _make_partition(tmp_path, "ocr_1", rows, with_images=True)

    assert nemotron.loose_image_dir(tmp_path, "ocr_1") is None


def test_a_partial_download_does_not_count_as_a_loose_image(tmp_path):
    """An interrupted fetch leaves .partial files; they are not images yet."""
    (tmp_path / "vqa_1_images").mkdir()
    (tmp_path / "vqa_1_images" / "0.jpg.partial").write_bytes(_JPEG)

    assert nemotron.loose_image_dir(tmp_path, "vqa_1") is None


def test_an_imageless_downloadable_partition_is_told_how_to_get_its_images(tmp_path):
    rows = [dict(_NEMOTRON_ROW, id="id0", image="0.png")]
    _make_partition(tmp_path, "vqa_1", rows, with_images=False)

    with pytest.raises(ValueError, match="download_nemotron_images.py"):
        nemotron.select_partitions(tmp_path, "vqa_1")


def test_fetch_one_writes_atomically_and_reports_a_404(tmp_path):
    """A removed key must not fail a run of a million, nor leave a stub file."""

    class _Response:
        def __init__(self, status: int, payload: bytes = b"") -> None:
            self.status = status
            self._payload = payload

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        def raise_for_status(self) -> None:
            if self.status >= 400:
                raise RuntimeError(self.status)

        async def read(self) -> bytes:
            return self._payload

    class _Session:
        def __init__(self, response: _Response) -> None:
            self._response = response

        def get(self, _url):
            return self._response

    async def run(response, destination):
        return await download.fetch_one(
            _Session(response),
            "https://example.invalid/x.jpg",
            destination,
            asyncio.Semaphore(1),
            max_retries=2,
        )

    good = tmp_path / "good.jpg"
    ok, num_bytes, error = asyncio.run(run(_Response(200, _JPEG), good))
    assert (ok, num_bytes, error) == (True, len(_JPEG), None)
    assert good.read_bytes() == _JPEG
    assert not list(tmp_path.glob("*.partial"))

    gone = tmp_path / "gone.jpg"
    ok, num_bytes, error = asyncio.run(run(_Response(404), gone))
    assert (ok, error) == (False, "404")
    assert not gone.exists()
