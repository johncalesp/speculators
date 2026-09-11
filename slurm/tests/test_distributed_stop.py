import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from cauldron_state import atomic_json
from train_cauldron import make_trainer


def _stop_worker(rank, directory):
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2
    )
    try:
        # Only rank 1 sees a local request; both ranks must pause together.
        os.environ["CAULDRON_STOP_FILE"] = str(root / f"stop-{rank}")
        trainer = make_trainer(object)()
        trainer.local_rank = "cpu"
        trainer.is_distributed = True
        requested = trainer._stop_requested()
        atomic_json(root / f"rank-{rank}.json", {"stop": requested})
    finally:
        dist.destroy_process_group()


def test_one_rank_stop_request_reaches_all_ranks(tmp_path):
    (tmp_path / "stop-1").touch()
    torch.multiprocessing.spawn(_stop_worker, args=(str(tmp_path),), nprocs=2)
    assert all(
        json.loads((tmp_path / f"rank-{rank}.json").read_text())["stop"]
        for rank in range(2)
    )
