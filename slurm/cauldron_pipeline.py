"""Run one bounded allocation of the persistent Cauldron workflow (exit 75 = resume)."""

import fcntl
import hashlib
import importlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from contextlib import suppress
from pathlib import Path

from cauldron_data import make_plan
from cauldron_state import atomic_json, recover_checkpoints, training_complete

REPO = Path(__file__).resolve().parents[1]
CONTINUE = 75


def env_int(name, default, minimum=1):
    value = int(os.environ.get(name, default))
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def source_paths():
    return sorted(
        [
            *REPO.joinpath("src").rglob("*.py"),
            *REPO.joinpath("hs_connectors/src").rglob("*.py"),
            *REPO.joinpath("scripts").glob("*.py"),
            *REPO.joinpath("slurm").glob("*.py"),
            *REPO.joinpath("slurm").glob("*.sh"),
        ]
    )


def source_digest():
    digest = hashlib.sha256()
    for path in source_paths():
        digest.update(str(path.relative_to(REPO)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def has_pipeline_work(root):
    if any(
        (root / name).exists()
        for name in ("prepared", "data_complete.json", "DONE.json")
    ):
        return True
    return any(
        directory.exists() and any(directory.iterdir())
        for directory in (root / "chunks", root / "checkpoints")
    )


def save_pipeline_config(root, config):
    path = root / "pipeline_config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        changed = [key for key in config if config[key] != previous.get(key)]
        if changed == ["source_digest"] and not has_pipeline_work(root):
            atomic_json(
                root / "provenance/config_revisions" / f"{time.time_ns()}.json",
                previous,
            )
            # Rebuild the plan if the updated source changes planning behavior.
            (root / "plan.json").unlink(missing_ok=True)
            print(
                "Source updated before data/training started; "
                "refreshing saved fingerprint.",
                flush=True,
            )
        elif changed:
            raise ValueError(
                f"Run configuration changed: {changed}. "
                "Restore it or use a new OUTPUT_DIR."
            )
        else:
            return
    atomic_json(path, config)


def initialize(root):
    from huggingface_hub import snapshot_download

    model = os.environ.get("MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
    model_path = Path(snapshot_download(model, local_files_only=True))
    source_override = os.environ.get("CAULDRON_DATA_DIR")
    source = (
        Path(source_override)
        if source_override
        else Path(
            snapshot_download(
                "HuggingFaceM4/the_cauldron", repo_type="dataset", local_files_only=True
            )
        )
    )
    config = {
        "dataset": "HuggingFaceM4/the_cauldron",
        "dataset_snapshot": str(source.absolute()),
        "model": model,
        "model_snapshot": str(model_path),
        "subsets": os.environ.get("CAULDRON_SUBSETS", "all"),
        "max_samples": env_int("MAX_SAMPLES", 0, 0),
        "chunk_size": env_int("CHUNK_SIZE", 128),
        "epochs": env_int("EPOCHS", 5),
        "seq_length": env_int("SEQ_LENGTH", 8192),
        "max_new_tokens": env_int("MAX_NEW_TOKENS", 512),
        "max_pixels": env_int("MAX_PIXELS", 1280 * 28 * 28),
        "max_images": env_int("MAX_IMAGES", 16),
        "temperature": float(os.environ.get("TEMPERATURE", "0")),
        "seed": env_int("SEED", 42, 0),
        "concurrency": env_int("REGEN_CONCURRENCY", 16),
        "num_layers": env_int("NUM_LAYERS", 5),
        "block_size": env_int("BLOCK_SIZE", 8),
        "max_anchors": env_int("MAX_ANCHORS", 512),
        "lr": float(os.environ.get("LR", "0.0003")),
        "checkpoint_steps": env_int("CHECKPOINT_STEPS", 200),
        "source_digest": source_digest(),
    }
    save_pipeline_config(root, config)
    if not (root / "plan.json").exists():
        plan = make_plan(
            source, config["subsets"], config["max_samples"], config["chunk_size"]
        )
        atomic_json(root / "plan.json", plan)
    return config


class Allocation:
    def __init__(self, root, config, deadline):
        self.root, self.config, self.deadline = root, config, deadline
        self.job = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
        self.provenance = root / "provenance" / self.job
        self.provenance.mkdir(parents=True, exist_ok=True)
        self.stop = self.provenance / "stop"
        self.paused = self.provenance / "training_paused.json"
        self.stop.unlink(missing_ok=True)
        self.paused.unlink(missing_ok=True)
        self.port = env_int("VLLM_PORT", 8000)
        self.endpoint = f"http://127.0.0.1:{self.port}"
        self.server = None
        self.devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        self.children = []
        self.logs = []
        self.expired = False
        self.hidden_states = Path(
            os.environ.get("HIDDEN_STATES_DIR", f"/tmp/dflash2-cauldron-{self.job}")
        )

    def save_provenance(self):
        atomic_json(
            self.provenance / "pipeline_command.json",
            {
                "argv": sys.argv,
                "config": self.config,
                "job_id": self.job,
                "deadline": self.deadline,
            },
        )
        shutil.copy(REPO / "slurm/training_script_cauldron.sh", self.provenance)
        for path in [*(REPO / "slurm").glob("*.py"), *(REPO / "slurm").glob("*.sh")]:
            shutil.copy(path, self.provenance)
        git_status = {"available": True}
        try:
            patch = subprocess.run(
                ["git", "diff", "HEAD", "--binary"],
                cwd=REPO,
                text=True,
                capture_output=True,
                check=True,
                timeout=30,
            ).stdout
        except (OSError, subprocess.SubprocessError) as error:
            detail = getattr(error, "stderr", None) or str(error)
            reason = " ".join(str(detail).split())[:2000]
            snapshot = self.save_source_snapshot()
            git_status = {
                "available": False,
                "error": reason,
                "source_snapshot": str(snapshot.relative_to(self.root)),
            }
            patch = f"# Git diff unavailable: {reason}\n# Source snapshot: {snapshot}\n"
            print(
                f"Warning: Git provenance unavailable ({reason}); "
                f"saved source snapshot to {snapshot}. Continuing.",
                flush=True,
            )
        atomic_json(self.provenance / "git_status.json", git_status)
        (self.provenance / "speculators.patch").write_text(patch)

    def save_source_snapshot(self):
        directory = self.root / "provenance/source_snapshots"
        directory.mkdir(parents=True, exist_ok=True)
        snapshot = directory / f"{self.config['source_digest']}.tar.gz"
        if not snapshot.exists():
            temporary = snapshot.with_suffix(".tmp")
            paths = source_paths() + [
                REPO / "pyproject.toml",
                REPO / "setup.py",
                REPO / "hs_connectors/pyproject.toml",
                REPO / "hs_connectors/setup.py",
            ]
            with tarfile.open(temporary, "w:gz", dereference=True) as archive:
                for path in paths:
                    if path.is_file():
                        archive.add(
                            path, arcname=str(path.relative_to(REPO)), recursive=False
                        )
            temporary.replace(snapshot)
        return snapshot

    def spawn(self, command, name, gpus=None, extra_env=None):
        env = dict(os.environ)
        if gpus is not None:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(
                self.devices[int(index)] for index in gpus.split(",")
            )
        env.update(extra_env or {})
        log_path = self.provenance / (name + ".log")
        stream = log_path.open("a")
        self.logs.append(stream)
        (self.provenance / (name + "_command.txt")).write_text(
            shlex.join(command) + "\n"
        )
        print(f"Starting {name}; log: {log_path}", flush=True)
        child = subprocess.Popen(
            command,
            env=env,
            cwd=REPO,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.children.append(child)
        return child

    def terminate(self, child, grace=60):
        if child is None or child not in self.children:
            return
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGTERM)
        with suppress(subprocess.TimeoutExpired):
            child.wait(timeout=grace)
        # The launcher can exit before its engine/worker descendants do.
        with suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        self.children.remove(child)

    def budget_expired(self):
        if self.expired or time.time() >= self.deadline:
            self.expired = True
            self.stop.touch()
        return self.expired

    def server_args(self, tp):
        cfg = self.config
        # vLLM 0.28 disables request logging by default; the old
        # --disable-log-requests flag is no longer accepted.
        return [
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(tp),
            "--dtype",
            "bfloat16",
            "--max-model-len",
            str(cfg["seq_length"] + cfg["max_new_tokens"]),
            "--max-num-seqs",
            "32",
            "--gpu-memory-utilization",
            "0.85",
            "--allowed-local-media-path",
            str(self.root / "chunks"),
            "--limit-mm-per-prompt",
            json.dumps({"image": cfg["max_images"]}),
            "--mm-processor-kwargs",
            json.dumps({"max_pixels": cfg["max_pixels"]}),
        ]

    def start_server(self, training):
        # Refuse to attach to a stale/unrelated service on the same port.
        import socket

        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                raise RuntimeError(
                    f"Port {self.port} is already occupied; set VLLM_PORT"
                )
        cfg = self.config
        if training:
            command = [
                sys.executable,
                "scripts/launch_vllm.py",
                "train",
                cfg["model"],
                "--provenance-dir",
                str(self.provenance / "vllm_training"),
                "--hidden-states-path",
                str(self.hidden_states),
                "--target-layer-ids",
                "2",
                "14",
                "25",
                "--",
                *self.server_args(2),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                cfg["model"],
                *self.server_args(4),
            ]
            # Normal generation must not enable hidden-state dumping.
            sys.path.insert(0, str(REPO / "scripts"))
            from launch_vllm import _save_vllm_provenance

            _save_vllm_provenance(
                command, str(self.provenance / "vllm_regeneration"), cfg["model"]
            )
        self.server = self.spawn(
            command,
            "vllm_training" if training else "vllm_regeneration",
            "0,1" if training else "0,1,2,3",
        )
        ready_by = time.time() + env_int("SERVER_START_TIMEOUT", 1200)
        while time.time() < ready_by:
            if self.budget_expired():
                return False
            if self.server.poll() is not None:
                raise RuntimeError("vLLM exited during startup; inspect its log")
            try:
                with urllib.request.urlopen(
                    self.endpoint + "/health", timeout=2
                ) as response:
                    if response.status == 200:
                        return True
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise TimeoutError("vLLM startup timed out")

    def supervise(self, child, training=False):
        stop_by = None
        while child.poll() is None:
            if self.server.poll() is not None:
                raise RuntimeError("vLLM exited while its client was running")
            if self.budget_expired() and stop_by is None:
                grace = env_int("SAVE_GRACE_SECONDS", 1200) if training else 120
                stop_by = time.time() + grace
                print(
                    "Time budget reached; requesting persistent progress and shutdown.",
                    flush=True,
                )
            if stop_by is not None and time.time() >= stop_by:
                self.terminate(child)
                # Unpublished data/checkpoint writes are ignored on restart.
                return CONTINUE
            time.sleep(2)
        if child.returncode != 0:
            raise RuntimeError(
                f"Worker exited with {child.returncode}; inspect its log"
            )
        self.terminate(child)
        return CONTINUE if self.expired else 0

    def run(self):
        cfg = self.config
        if training_complete(self.root / "checkpoints", cfg["epochs"]):
            return 0
        if not (self.root / "prepared").exists():
            if not self.start_server(training=False):
                return CONTINUE
            worker = self.spawn(
                [
                    sys.executable,
                    "slurm/cauldron_data.py",
                    "--output",
                    str(self.root),
                    "--endpoint",
                    self.endpoint,
                    "--stop-file",
                    str(self.stop),
                ],
                "prepare",
            )
            result = self.supervise(worker)
            self.terminate(self.server)
            self.server = None
            if result == CONTINUE:
                return result
            if not (self.root / "prepared").exists():
                raise RuntimeError(
                    "Data worker exited without publishing a prepared dataset"
                )
        if self.budget_expired() or not self.start_server(training=True):
            return CONTINUE
        self.hidden_states.mkdir(parents=True, exist_ok=True)
        train = self.spawn(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                "slurm/train_cauldron.py",
                "--verifier-name-or-path",
                cfg["model"],
                "--data-path",
                str(self.root / "prepared"),
                "--save-path",
                str(self.root / "checkpoints"),
                "--vllm-endpoint",
                self.endpoint + "/v1",
                "--hidden-states-path",
                str(self.hidden_states),
                "--on-missing",
                "generate",
                "--on-generate",
                "delete",
                "--speculator-type",
                "dflash2",
                "--epochs",
                str(cfg["epochs"]),
                "--lr",
                str(cfg["lr"]),
                "--total-seq-len",
                str(cfg["seq_length"]),
                "--num-layers",
                str(cfg["num_layers"]),
                "--block-size",
                str(cfg["block_size"]),
                "--max-anchors",
                str(cfg["max_anchors"]),
                "--target-layer-ids",
                "2",
                "14",
                "25",
                "--checkpoint-freq",
                "1",
                "--log-freq",
                "20",
                "--fsdp-shard",
                "--gradient-checkpointing",
            ],
            "train",
            "2,3",
            {
                "CAULDRON_STOP_FILE": str(self.stop),
                "CAULDRON_PAUSED_FILE": str(self.paused),
                "CAULDRON_CHECKPOINT_STEPS": str(cfg["checkpoint_steps"]),
            },
        )
        result = self.supervise(train, training=True)
        if training_complete(self.root / "checkpoints", cfg["epochs"]):
            return 0
        if result == CONTINUE or self.paused.exists():
            return CONTINUE
        raise RuntimeError("Trainer exited before the requested epochs were committed")

    def close(self):
        for child in reversed(list(self.children)):
            self.terminate(child)
        for stream in self.logs:
            stream.close()


def progress(root):
    complete_chunks = list((root / "chunks").glob("*/complete.json"))
    latest = []
    for path in (root / "checkpoints").glob("[0-9]*/training_state.json"):
        state = json.loads(path.read_text())
        latest.append(state["global_step"])
    # Include partial chunks: completed conversations are useful progress.
    journals = sum(
        path.stat().st_size for path in (root / "chunks").glob("*/regenerated.jsonl")
    )
    return [len(complete_chunks), journals, max(latest, default=0)]


def main():
    root = Path(os.environ["OUTPUT_DIR"]).absolute()
    root.mkdir(parents=True, exist_ok=True)
    deadline = int(os.environ.get("CAULDRON_DEADLINE", int(time.time()) + 16200))
    with (root / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "DONE.json").exists():
            print(f"Pipeline already complete: {root / 'DONE.json'}")
            return 0
        # Validate the container before reserving hours of unsuccessful work.
        import torch

        importlib.import_module("vllm")
        importlib.import_module("speculators.models.dflash2")
        from transformers import AutoConfig

        if torch.cuda.device_count() != 4:
            raise RuntimeError("This workflow requires exactly four visible GPUs")
        cfg = initialize(root)
        verifier = AutoConfig.from_pretrained(cfg["model"])
        text = verifier.get_text_config()
        if text.num_hidden_layers != 28 or text.vocab_size != 152064:
            raise ValueError(
                "This recipe targets Qwen2.5-VL-7B; layer IDs/vocabulary do not match"
            )
        recover_checkpoints(root / "checkpoints")
        before = progress(root)
        allocation = Allocation(root, cfg, deadline)
        try:
            allocation.save_provenance()
            result = allocation.run()
        finally:
            allocation.close()
        if result == 0:
            atomic_json(
                root / "DONE.json",
                {
                    "epochs": cfg["epochs"],
                    "checkpoint": str(root / "checkpoints" / str(cfg["epochs"] - 1)),
                    "finished_at": time.time(),
                },
            )
            print(f"Training complete: {root / 'DONE.json'}")
        else:
            after = progress(root)
            path = root / "continuation.json"
            old = json.loads(path.read_text()) if path.exists() else {}
            stalled = old.get("stalled_allocations", 0) + 1 if after == before else 0
            atomic_json(path, {"progress": after, "stalled_allocations": stalled})
            if stalled >= 3:
                raise RuntimeError(
                    "Three allocations made no progress; "
                    "inspect logs before resubmitting"
                )
            print(
                f"Progress saved at {root}; another allocation is needed.", flush=True
            )
        return result


if __name__ == "__main__":
    sys.exit(main())
