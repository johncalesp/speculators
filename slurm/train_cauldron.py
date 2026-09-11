"""Slurm Trainer adapter; the public training CLI stays unchanged.

The scheduler hook runs after the optimizer update on every rank. Pause there,
after advancing the scheduler, before any rank starts the next training step.
The upstream loop increments global_step after this hook; a paused step must
do that itself. Keep this contract covered when updating Trainer.
"""

import os
import shutil
from pathlib import Path

from cauldron_state import atomic_json, publish_checkpoint


class TrainingPausedError(Exception):
    pass


class CauldronTrainerMixin:
    def _barrier(self):
        if self.is_distributed:
            import torch.distributed as dist

            dist.barrier()

    def _stop_requested(self):
        stop = Path(os.environ["CAULDRON_STOP_FILE"]).exists()
        if self.is_distributed:
            import torch
            import torch.distributed as dist

            value = torch.tensor(int(stop), device=self.local_rank)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            stop = bool(value.item())
        return stop

    def _prepare_resume_skip(self, epoch):
        skipped = super()._prepare_resume_skip(epoch)
        self._cauldron_epoch = epoch
        self._cauldron_step = skipped
        return skipped

    def _schedulers_step(self):
        super()._schedulers_step()
        self._cauldron_step += 1
        stop = self._stop_requested()
        interval = int(os.environ.get("CAULDRON_CHECKPOINT_STEPS", "200"))
        if stop or (self.global_step + 1) % interval == 0:
            self.global_step += 1
            self.maybe_save_checkpoint(
                self._cauldron_epoch, local_step=self._cauldron_step
            )
            if stop:
                raise TrainingPausedError
            # The ordinary loop will increment this after its logging.
            self.global_step -= 1

    def _maybe_val_sync(self, batch_index):
        super()._maybe_val_sync(batch_index)
        if self._stop_requested():
            # Training's end-of-epoch checkpoint already exists.
            raise TrainingPausedError

    def maybe_save_checkpoint(self, epoch, local_step=0):
        if not isinstance(epoch, int):
            # A signal can interrupt a collective/update. Retain the last
            # committed checkpoint instead of publishing uncertain state.
            self._cauldron_interrupted = True
            return
        root = self.checkpointer.path
        label = f".{epoch}.incomplete"
        staging = root / label
        if self.rank == 0:
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
        self._barrier()
        self.checkpointer.save_checkpoint(self.model, self.optimizers, label)
        if self.schedulers:
            self.checkpointer.save_scheduler_state_dict(self.schedulers, label)
        if self.rank == 0:
            atomic_json(
                staging / "training_state.json",
                {
                    "epoch": epoch,
                    "local_step": local_step,
                    "global_step": self.global_step,
                },
            )
            publish_checkpoint(root, epoch)
            print(
                f"Committed epoch={epoch} step={local_step} global={self.global_step}",
                flush=True,
            )
        self._barrier()

    def run_training(self):
        if self.config.save_best or self.config.max_steps is not None:
            raise ValueError("Cauldron requires save_best=False and no max_steps")
        try:
            super().run_training()
        except TrainingPausedError:
            if self.rank == 0:
                atomic_json(
                    Path(os.environ["CAULDRON_PAUSED_FILE"]),
                    {
                        "global_step": self.global_step,
                    },
                )
            return
        if getattr(self, "_cauldron_interrupted", False):
            raise RuntimeError(
                "Unexpected training signal; last committed checkpoint retained"
            )


def make_trainer(base):
    return type("CauldronTrainer", (CauldronTrainerMixin, base), {})


def main():
    from speculators.train import cli
    from speculators.train.config import TrainConfig

    cli.Trainer = make_trainer(cli.Trainer)
    cli.main(TrainConfig.resolve())


if __name__ == "__main__":
    main()
