import json
import subprocess
import tarfile

import pytest

import cauldron_pipeline as pipeline
from cauldron_state import atomic_json


@pytest.mark.parametrize("failure", ["missing", "failed", "timeout", None])
def test_provenance_retains_sources_when_git_is_unavailable(
    tmp_path, monkeypatch, capsys, failure
):
    repo = tmp_path / "repo"
    for name, text in {
        "slurm/training_script_cauldron.sh": "#!/bin/bash\n",
        "slurm/run_cauldron.sh": "#!/bin/bash\n",
        "src/example.py": "value = 42\n",
        "pyproject.toml": "[project]\nname = 'example'\n",
    }.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(pipeline, "REPO", repo)

    def run_git(command, **kwargs):
        assert command == ["git", "diff", "HEAD", "--binary"]
        assert kwargs["timeout"] == 30
        if failure == "missing":
            raise FileNotFoundError(2, "No such file or directory", "git")
        if failure == "failed":
            raise subprocess.CalledProcessError(
                128, command, stderr="not a git repository"
            )
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 30)
        return subprocess.CompletedProcess(command, 0, stdout="diff --git a/x b/x\n")

    monkeypatch.setattr(pipeline.subprocess, "run", run_git)
    allocation = pipeline.Allocation(
        tmp_path / "output", {"source_digest": pipeline.source_digest()}, 9999999999
    )
    allocation.save_provenance()
    status = json.loads((allocation.provenance / "git_status.json").read_text())
    patch = (allocation.provenance / "speculators.patch").read_text()
    assert (allocation.provenance / "pipeline_command.json").is_file()
    assert (allocation.provenance / "run_cauldron.sh").is_file()
    if failure:
        assert status["available"] is False
        assert "Git diff unavailable" in patch
        with tarfile.open(allocation.root / status["source_snapshot"]) as archive:
            assert archive.extractfile("src/example.py").read() == b"value = 42\n"
            assert archive.extractfile("pyproject.toml").read().startswith(b"[project]")
        assert "Continuing" in capsys.readouterr().out
    else:
        assert status == {"available": True}
        assert patch == "diff --git a/x b/x\n"
        assert not (allocation.root / "provenance/source_snapshots").exists()


def test_source_fix_can_restart_a_run_that_failed_before_work(tmp_path):
    previous = {"source_digest": "before", "epochs": 5}
    current = {"source_digest": "after", "epochs": 5}
    atomic_json(tmp_path / "pipeline_config.json", previous)
    atomic_json(tmp_path / "plan.json", [{"id": "stale"}])
    (tmp_path / "checkpoints").mkdir()  # Created before save_provenance failed.
    pipeline.save_pipeline_config(tmp_path, current)
    assert json.loads((tmp_path / "pipeline_config.json").read_text()) == current
    archives = list((tmp_path / "provenance/config_revisions").glob("*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == previous
    assert not (tmp_path / "plan.json").exists()


@pytest.mark.parametrize(
    "artifact",
    [
        "chunks/0000000/conversations.json",
        "checkpoints/0/training_state.json",
        "checkpoints/.0.incomplete/model.safetensors",
        "prepared/state.json",
        "data_complete.json",
        "DONE.json",
    ],
)
def test_source_fix_cannot_change_a_run_with_work(tmp_path, artifact):
    previous = {"source_digest": "before", "epochs": 5}
    atomic_json(tmp_path / "pipeline_config.json", previous)
    atomic_json(tmp_path / artifact, {})
    with pytest.raises(ValueError, match="source_digest"):
        pipeline.save_pipeline_config(tmp_path, {**previous, "source_digest": "after"})
    assert json.loads((tmp_path / "pipeline_config.json").read_text()) == previous


def test_prework_retry_still_rejects_training_setting_changes(tmp_path):
    previous = {"source_digest": "before", "epochs": 5}
    atomic_json(tmp_path / "pipeline_config.json", previous)
    with pytest.raises(ValueError, match="epochs"):
        pipeline.save_pipeline_config(
            tmp_path, {"source_digest": "after", "epochs": 10}
        )
    assert json.loads((tmp_path / "pipeline_config.json").read_text()) == previous
