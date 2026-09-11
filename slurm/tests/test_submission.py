import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("worker_exit", "auto", "expected_exit", "resubmit"),
    [(0, "1", 0, False), (1, "1", 1, False), (75, "1", 0, True), (75, "0", 0, False)],
)
def test_batch_submission_only_chains_continuations(
    tmp_path, worker_exit, auto, expected_exit, resubmit
):
    binary = tmp_path / "bin"
    binary.mkdir()
    for name, contents in {
        "srun": '#!/bin/bash\nexit "$TEST_WORKER_EXIT"\n',
        "sbatch": '#!/bin/bash\nprintf "%s\\n" "$@" > "$TEST_SUBMISSION"\necho 9002\n',
    }.items():
        path = binary / name
        path.write_text(contents)
        path.chmod(0o755)
    captured = tmp_path / "submission"
    env = dict(os.environ)
    env.update(
        {
            "PATH": f"{binary}:{env['PATH']}",
            "TEST_WORKER_EXIT": str(worker_exit),
            "TEST_SUBMISSION": str(captured),
            "AUTO_CONTINUE": auto,
            "SLURM_JOB_ID": "9001",
            "SLURM_SUBMIT_DIR": str(REPO),
            "SLURM_JOB_PARTITION": "test-partition",
            "SLURM_JOB_ACCOUNT": "test-account",
        }
    )
    result = subprocess.run(  # noqa: S603 - local fake Slurm commands
        [str(Path("/bin/bash")), str(REPO / "slurm/training_script_cauldron.sh")],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == expected_exit, result.stderr
    assert captured.exists() == resubmit
    if resubmit:
        args = captured.read_text().splitlines()
        assert "--dependency=afterany:9001" in args
        assert "--partition=test-partition" in args
        assert "--account=test-account" in args
        assert args[-1] == str(REPO / "slurm/training_script_cauldron.sh")
        assert "Continuation job: 9002" in result.stdout


@pytest.mark.parametrize("failure", [None, "datasets", "editable", "preflight"])
def test_container_bootstrap_installs_before_pipeline(tmp_path, failure):
    interpreter = tmp_path / "python3"
    interpreter.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['TEST_PYTHON_CALLS'], 'a') as stream:\n"
        "    stream.write(json.dumps({'args': args, 'cwd': os.getcwd(),\n"
        "        'cache': os.environ['PIP_CACHE_DIR']}) + '\\n')\n"
        "if args[:3] == ['-m', 'pip', 'install']:\n"
        "    phase = 'editable' if '--no-deps' in args else 'datasets'\n"
        "elif args[0] == '-c' and args[1].startswith('import datasets'):\n"
        "    phase = 'preflight'\n"
        "elif args == ['slurm/cauldron_pipeline.py']:\n"
        "    sys.exit(75)\n"
        "else:\n"
        "    phase = 'version'\n"
        # Even an installer returning 75 must not trigger job continuation.
        "sys.exit(75 if phase == os.environ.get('TEST_FAIL_PHASE') else 0)\n"
    )
    interpreter.chmod(0o755)
    captured = tmp_path / "python_calls.jsonl"
    env = dict(os.environ)
    env.pop("PIP_CACHE_DIR", None)
    env.update(
        {
            "PYTHON": str(interpreter),
            "HF_HOME": str(tmp_path / "cache"),
            "TEST_PYTHON_CALLS": str(captured),
            "TEST_FAIL_PHASE": failure or "",
        }
    )
    result = subprocess.run(  # noqa: S603 - fake interpreter; no real installs
        ["/bin/bash", str(REPO / "slurm/run_cauldron.sh")],
        env=env,
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    calls = [json.loads(line) for line in captured.read_text().splitlines()]
    commands = [call["args"] for call in calls]
    assert commands[1] == ["-m", "pip", "install", "datasets>=4.0.0,<=5.0.1"]
    assert all(call["cwd"] == str(REPO) for call in calls)
    assert all(call["cache"] == str(tmp_path / "cache/pip") for call in calls)
    if failure:
        assert result.returncode == 1
        assert ["slurm/cauldron_pipeline.py"] not in commands
        assert "failed" in result.stderr
    else:
        assert commands[2] == [
            "-m",
            "pip",
            "install",
            "--no-deps",
            "-e",
            "./hs_connectors",
            "-e",
            ".",
        ]
        assert commands[3] == [
            "-c",
            "import datasets, hs_connectors, speculators, speculators.train.data",
        ]
        assert commands[4] == ["slurm/cauldron_pipeline.py"]
        assert result.returncode == 75
