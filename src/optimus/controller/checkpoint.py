"""
Optimus Checkpoint Manager.
Handles atomic saving and restoration of model and optimizer checkpoints
on shared storage, maintains latest.json pointer, and prunes older commits.
"""

from dataclasses import asdict, dataclass
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional
import torch

logger = logging.getLogger("optimus.controller.checkpoint")


@dataclass
class CheckpointMetadata:
    job_id: str
    epoch: int
    global_step: int
    dataset_id: str
    dataset_version: str
    world_size: int
    loss: float
    checkpoint_dir: str


class CheckpointManager:
    """Manages atomic checkpoints for a training job."""

    def __init__(
        self,
        base_dir: str = "/srv/checkpoints",
        job_id: str = "default_job",
        max_retained: int = 3,
    ):
        self.job_dir = os.path.join(os.path.abspath(base_dir), job_id)
        self.job_id = job_id
        self.max_retained = max_retained
        os.makedirs(self.job_dir, exist_ok=True)
        self.latest_pointer_file = os.path.join(self.job_dir, "latest.json")

    def get_latest_checkpoint_dir(self) -> Optional[str]:
        if not os.path.isfile(self.latest_pointer_file):
            return None
        try:
            with open(self.latest_pointer_file, "r") as f:
                data = json.load(f)
            path = data.get("latest_checkpoint")
            if path and os.path.isdir(path):
                return path
        except Exception as e:
            logger.warning(f"Failed to read latest checkpoint pointer: {e}")
        return None

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        dataset_id: str,
        dataset_version: str,
        world_size: int,
        loss: float = 0.0,
        lr_scheduler: Optional[Any] = None,
    ) -> str:
        """
        Atomically saves training state into checkpoint-<epoch> and updates latest.json.
        Only executed by Rank 0 in the distributed training group.
        """
        final_dir = os.path.join(self.job_dir, f"checkpoint-{epoch:04d}")
        temp_dir = f"{final_dir}.tmp.{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Save model weights
            # If wrapped in DistributedDataParallel, save module.state_dict()
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, os.path.join(temp_dir, "model.pt"))

            # Save optimizer state
            torch.save(optimizer.state_dict(), os.path.join(temp_dir, "optimizer.pt"))

            # Save LR scheduler if provided
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join(temp_dir, "scheduler.pt"))

            # Save RNG states
            rng_states = {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            }
            torch.save(rng_states, os.path.join(temp_dir, "rng.pt"))

            # Save metadata
            meta = CheckpointMetadata(
                job_id=self.job_id,
                epoch=epoch,
                global_step=global_step,
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                world_size=world_size,
                loss=loss,
                checkpoint_dir=final_dir,
            )
            with open(os.path.join(temp_dir, "metadata.json"), "w") as f:
                json.dump(asdict(meta), f, indent=2)

            # Atomically rename
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            os.replace(temp_dir, final_dir)

            # Atomically update latest.json pointer
            temp_pointer = f"{self.latest_pointer_file}.tmp.{os.getpid()}"
            with open(temp_pointer, "w") as f:
                json.dump(
                    {
                        "latest_checkpoint": final_dir,
                        "epoch": epoch,
                        "step": global_step,
                    },
                    f,
                    indent=2,
                )
            os.replace(temp_pointer, self.latest_pointer_file)

            logger.info(f"Committed atomic checkpoint for epoch {epoch} -> '{final_dir}'")
            self._prune_older_checkpoints()
            return final_dir

        except Exception as e:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise RuntimeError(f"Failed to commit checkpoint for epoch {epoch}: {e}")

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[Any] = None,
    ) -> CheckpointMetadata:
        """Restores model and optimizer states from a checkpoint directory."""
        model_file = os.path.join(checkpoint_dir, "model.pt")
        opt_file = os.path.join(checkpoint_dir, "optimizer.pt")
        meta_file = os.path.join(checkpoint_dir, "metadata.json")

        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Checkpoint model weights not found in '{checkpoint_dir}'")

        state_dict = torch.load(model_file, map_location="cpu")
        target_model = model.module if hasattr(model, "module") else model
        target_model.load_state_dict(state_dict)

        if optimizer and os.path.isfile(opt_file):
            opt_state = torch.load(opt_file, map_location="cpu")
            optimizer.load_state_dict(opt_state)

        if lr_scheduler and os.path.isfile(os.path.join(checkpoint_dir, "scheduler.pt")):
            sched_state = torch.load(os.path.join(checkpoint_dir, "scheduler.pt"), map_location="cpu")
            lr_scheduler.load_state_dict(sched_state)

        with open(meta_file, "r") as f:
            data = json.load(f)

        logger.info(f"Restored checkpoint from '{checkpoint_dir}' (Epoch: {data['epoch']})")
        return CheckpointMetadata(**data)

    def _prune_older_checkpoints(self) -> None:
        """Retains only the latest max_retained checkpoint directories."""
        entries = []
        for name in os.listdir(self.job_dir):
            if name.startswith("checkpoint-") and not name.endswith(".tmp"):
                full_path = os.path.join(self.job_dir, name)
                if os.path.isdir(full_path):
                    entries.append(full_path)

        entries.sort()
        while len(entries) > self.max_retained:
            oldest = entries.pop(0)
            try:
                shutil.rmtree(oldest)
                logger.info(f"Pruned older checkpoint: {oldest}")
            except Exception as e:
                logger.warning(f"Failed to prune {oldest}: {e}")
