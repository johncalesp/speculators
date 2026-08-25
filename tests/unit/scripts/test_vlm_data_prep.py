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
import json
from pathlib import Path
from typing import Any

import pytest

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
regen = _load_script(
    "regenerate_vlm_responses", _SCRIPTS_DIR / "regenerate_vlm_responses.py"
)


# A minimal but real PNG header, enough for suffix sniffing.
_PNG = b"\x89PNG\r\n\x1a\n" + b"pixels"
_JPEG = b"\xff\xd8\xff" + b"pixels"

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
    rows = [
        {**_exported_row(tmp_path), "conversation_id": f"c{i}"} for i in range(2)
    ]
    data = tmp_path / "prompts.jsonl"
    data.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
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
