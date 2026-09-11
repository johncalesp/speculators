import os
import subprocess
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
