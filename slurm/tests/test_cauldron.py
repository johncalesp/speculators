import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from datasets import load_from_disk

import cauldron_data as data
from cauldron_state import (
    append_journal,
    atomic_json,
    publish_checkpoint,
    read_journal,
    recover_checkpoints,
    training_complete,
)
from train_cauldron import TrainingPausedError, make_trainer


def source_dataset(root, subsets=("a", "b"), rows=3):
    root.mkdir()
    (root / "README.md").write_text(
        "---\nconfigs:\n"
        + "".join(f"- config_name: {name}\n" for name in subsets)
        + "---\n"
    )
    for name in subsets:
        directory = root / name
        directory.mkdir()
        pq.write_table(
            pa.Table.from_pylist(
                [
                    {
                        "images": [{"bytes": b"image-bytes", "path": None}],
                        "texts": [
                            {"user": f"question {index}", "assistant": "source answer"}
                        ],
                    }
                    for index in range(rows)
                ]
            ),
            directory / "train-00000-of-00001.parquet",
        )
    return root


def test_plan_limits_and_checks_all_cached_subsets(tmp_path):
    source = source_dataset(tmp_path / "source")
    plan = data.make_plan(source, "all", 5, 2)
    assert sum(task["count"] for task in plan) == 5
    assert [task["subset"] for task in plan] == ["a", "b", "a"]
    (source / "README.md").write_text("---\nconfigs:\n- config_name: missing\n---\n")
    with pytest.raises(ValueError, match="Missing cached"):
        data.make_plan(source, "all", 0, 2)


def test_journal_discards_only_torn_tail(tmp_path):
    journal = tmp_path / "rows.jsonl"
    append_journal(journal, {"id": "a"})
    with journal.open("ab") as stream:
        stream.write(b'{"id": "unfinished')
    assert read_journal(journal) == [{"id": "a"}]
    append_journal(journal, {"id": "b"})
    assert [row["id"] for row in read_journal(journal)] == ["a", "b"]
    journal.write_bytes(b"broken\n")
    with pytest.raises(json.JSONDecodeError):
        read_journal(journal)


def fake_sample(row):
    from speculators.data_generation.preprocessing import _adapt_conv_for_vllm

    messages = _adapt_conv_for_vllm(row["conversations"])
    return {
        "id": row["id"],
        "skipped": None,
        "conversations": [],
        "samples": [
            {
                "input_ids": [1, 2, 3],
                "loss_mask": [0, 1, 1],
                "seq_len": 3,
                "messages": messages,
            }
        ],
    }


def test_data_resume_preserves_images_and_composes_arrow(tmp_path, monkeypatch):
    source = source_dataset(tmp_path / "source", rows=2)
    output = tmp_path / "output"
    plan = data.make_plan(source, "all", 0, 2)
    config = {"concurrency": 1, "seed": 42}
    stop = tmp_path / "stop"
    calls = []

    def regenerate(row, *_):
        calls.append(row["id"])
        stop.touch()
        return fake_sample(row)

    monkeypatch.setattr(data, "regenerate", regenerate)
    data.prepare_chunk(plan[0], config, output, "unused", stop)
    assert len(calls) == 1
    assert not (output / "chunks/0000000/complete.json").exists()
    stop.unlink()
    monkeypatch.setattr(
        data,
        "regenerate",
        lambda row, *_: (calls.append(row["id"]), fake_sample(row))[1],
    )
    for task in plan:
        data.prepare_chunk(task, config, output, "unused", stop)
    assert len(calls) == len(set(calls)) == 4
    # Crash after publishing prepared Arrow but before its completion marker.
    (output / "chunks/0000000/complete.json").unlink()
    data.prepare_chunk(plan[0], config, output, "unused", stop)
    assert len(calls) == 4
    data.assemble(output, plan, 42)
    dataset = load_from_disk(str(output / "prepared"))
    assert len(dataset) == 4
    assert isinstance(dataset[0]["input_ids"], torch.Tensor)
    image = dataset[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert image.startswith("file://")
    assert Path(image.removeprefix("file://")).read_bytes() == b"image-bytes"
    assert all(path.is_symlink() for path in (output / "prepared").glob("*.arrow"))


def test_regeneration_uses_images_and_own_multiturn_history(monkeypatch):
    import speculators.data_generation.render_client as rendering
    from speculators.data_generation import preprocessing

    def render(_endpoint, messages, *, add_generation_prompt, **_):
        tokens = []
        for message in messages:
            tokens.append(10 if message["role"] == "user" else 20)
            for part in message["content"]:
                tokens.extend(ord(c) for c in part["text"]) if part[
                    "type"
                ] == "text" else tokens.append(99)
            if message["role"] == "assistant":
                tokens.append(30)
        if add_generation_prompt:
            tokens.append(20)
        return tokens

    requests = []

    def chat(_endpoint, payload):
        requests.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {"content": "new"}}]}

    monkeypatch.setattr(data, "post_chat", chat)
    monkeypatch.setattr(rendering, "render_conversation", render)
    monkeypatch.setattr(preprocessing, "render_conversation", render)
    row = {
        "id": "one",
        "num_images": 1,
        "conversations": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": "/tmp/photo.png"},
                    {"type": "text", "text": "first"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "source answer"}],
            },
            {"role": "user", "content": [{"type": "text", "text": "next"}]},
        ],
    }
    cfg = {
        "model": "test",
        "max_images": 16,
        "seq_length": 256,
        "max_new_tokens": 32,
        "temperature": 0,
        "seed": 42,
    }
    result = data.regenerate(row, cfg, "unused")
    assert len(result["samples"]) == 2
    assert (
        requests[0]["messages"][0]["content"][0]["image_url"]["url"]
        == "file:///tmp/photo.png"
    )
    assert requests[1]["messages"][1]["content"][0]["text"] == "new"
    assert all(sum(sample["loss_mask"]) == 4 for sample in result["samples"])


@pytest.mark.parametrize("during_publish", [False, True])
def test_checkpoint_recovers_interrupted_replacement(tmp_path, during_publish):
    (tmp_path / "0").mkdir()
    (tmp_path / "0/model").write_text("old")
    (tmp_path / ".0.incomplete").mkdir()
    (tmp_path / ".0.incomplete/model").write_text("partial")
    if during_publish:
        (tmp_path / "0").rename(tmp_path / ".0.previous")
    recover_checkpoints(tmp_path)
    assert (tmp_path / "0/model").read_text() == "old"
    assert not (tmp_path / ".0.incomplete").exists()
    (tmp_path / ".0.incomplete").mkdir()
    (tmp_path / ".0.incomplete/model").write_text("new")
    publish_checkpoint(tmp_path, 0)
    assert (tmp_path / "0/model").read_text() == "new"
    assert not (tmp_path / ".0.previous").exists()


def test_pause_uses_real_training_loop_and_resumes_next_batch(tmp_path, monkeypatch):
    from speculators.train.trainer import Trainer, TrainerConfig

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.seen = []

        def forward(self, input_ids, **_):
            self.seen.append(input_ids.item())
            return None, self.weight.square(), {}

    class Samples(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor(index),
                "error_records": 0,
                "document_ids": torch.tensor([0]),
            }

    class Sampler:
        def __init__(self):
            self._cached_generated_batches = None

        def _generate_batches(self, epoch):
            return [[0], [1], [2], [3]]

        def __len__(self):
            return len(list(self.__iter__()))

        def __iter__(self):
            if self._cached_generated_batches is not None:
                return iter(self._cached_generated_batches[1])
            return iter(self._generate_batches(0))

    stop = tmp_path / "stop"
    stop.touch()
    monkeypatch.setenv("CAULDRON_STOP_FILE", str(stop))
    monkeypatch.setenv("CAULDRON_CHECKPOINT_STEPS", "2")
    cls = make_trainer(Trainer)
    trainer = cls.__new__(cls)
    trainer.config = TrainerConfig(
        lr=0.01, num_epochs=1, save_path=str(tmp_path), log_freq=20
    )
    trainer.model = Model()
    trainer.local_rank, trainer.device_type, trainer.rank = "cpu", "cpu", 1
    trainer.is_distributed = False
    trainer.current_epoch, trainer.global_step = 0, 1
    trainer.train_loader = torch.utils.data.DataLoader(
        Samples(), batch_sampler=Sampler()
    )
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=0.01)]
    trainer.schedulers = [torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 1)]
    saves = []
    trainer.maybe_save_checkpoint = lambda epoch, local_step=0: saves.append(
        (epoch, local_step, trainer.global_step, trainer.schedulers[0].last_epoch)
    )
    with pytest.raises(TrainingPausedError):
        trainer.train_epoch(0)
    assert saves == [(0, 1, 2, 1)]
    assert trainer.model.seen == [0]
    stop.unlink()
    trainer._resume_local_step = 1
    trainer.train_epoch(0)
    assert trainer.model.seen == [0, 1, 2, 3]
    assert trainer.global_step == 5
    assert trainer.schedulers[0].last_epoch == 4
    assert saves[-1] == (0, 3, 4, 3)


def test_training_complete_requires_final_epoch_boundary(tmp_path):
    atomic_json(tmp_path / "4/training_state.json", {"epoch": 4, "local_step": 10})
    assert not training_complete(tmp_path, 5)
    atomic_json(tmp_path / "4/training_state.json", {"epoch": 4, "local_step": 0})
    assert training_complete(tmp_path, 5)


def test_training_command_resolves_with_current_cli(tmp_path, monkeypatch):
    import huggingface_hub

    from cauldron_pipeline import Allocation, initialize
    from speculators.train.config import TrainConfig

    source = source_dataset(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: str(source)
    )
    monkeypatch.setenv("CAULDRON_SUBSETS", "all")
    config = initialize(output)
    (output / "prepared").mkdir()
    allocation = Allocation(output, config, 9999999999)
    monkeypatch.setattr(allocation, "start_server", lambda **kwargs: True)
    commands = []
    monkeypatch.setattr(
        allocation, "spawn", lambda command, *a, **k: commands.append(command)
    )

    def supervise(*args, **kwargs):
        atomic_json(
            output / "checkpoints/4/training_state.json", {"epoch": 4, "local_step": 0}
        )
        return 0

    monkeypatch.setattr(allocation, "supervise", supervise)
    assert allocation.run() == 0
    command = commands[0]
    args = command[command.index("slurm/train_cauldron.py") + 1 :]
    parsed = TrainConfig.resolve(args).flatten()
    assert parsed["speculator_type"] == "dflash2"
    assert parsed["target_layer_ids"] == [2, 14, 25]
    assert parsed["epochs"] == 5
    assert parsed["draft_vocab_size"] is None
    assert parsed["fsdp_shard"] is True


def test_published_checkpoint_restores_model_optimizer_scheduler(tmp_path, monkeypatch):
    from transformers import PretrainedConfig, PreTrainedModel

    from speculators.train import checkpointer

    class TinyModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self):
            super().__init__(PretrainedConfig())
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    monkeypatch.setattr(checkpointer, "get_current_device", lambda: "cpu")
    trainer = make_trainer(object)()
    trainer.rank = 0
    trainer.is_distributed = False
    trainer.global_step = 1
    trainer.model = TinyModel()
    trainer.optimizers = [torch.optim.AdamW(trainer.model.parameters(), lr=0.01)]
    trainer.schedulers = [torch.optim.lr_scheduler.StepLR(trainer.optimizers[0], 1)]
    trainer.model.weight.square().sum().backward()
    trainer.optimizers[0].step()
    trainer.schedulers[0].step()
    trainer.checkpointer = checkpointer.SingleGPUCheckpointer(str(tmp_path))
    trainer.maybe_save_checkpoint(0, local_step=1)

    restored = checkpointer.SingleGPUCheckpointer(str(tmp_path))
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    restored.load_model_state_dict(model)
    restored.load_optimizer_state_dict(model, optimizer)
    restored.load_scheduler_state_dict(scheduler)
    assert torch.allclose(model.weight, trainer.model.weight, atol=0.004)
    assert optimizer.state[model.weight]["step"].item() == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.001)
    assert scheduler.last_epoch == 1
    assert json.loads((tmp_path / "0/training_state.json").read_text()) == {
        "epoch": 0,
        "local_step": 1,
        "global_step": 1,
    }
