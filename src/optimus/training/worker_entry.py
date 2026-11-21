"""
Optimus Worker Training Process Entrypoint.
Standard execution script spawned by optimusd (the worker daemon) to run DDP training.
Initializes Gloo process group, runs epoch batches, synchronizes gradients,
and allows Rank 0 to write atomic checkpoints.
"""

import argparse
import inspect
import json
import logging
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader

from optimus.controller.checkpoint import CheckpointManager
from optimus.training.ddp import cleanup_distributed, init_distributed
from optimus.training.factories import (
    DEFAULT_DATASET_FACTORY,
    DEFAULT_MODEL_FACTORY,
    accepts_kwarg,
    model_factory_path_from_name,
    resolve_factory,
)
from optimus.training.sampler import OptimusDistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Rank %(rank)s] %(message)s",
)


def get_logger(rank: int):
    logger = logging.getLogger("optimus.worker_entry")
    extra = {"rank": str(rank)}
    return logging.LoggerAdapter(logger, extra)


def update_status_file(
    logs_dir: str,
    job_id: str,
    epoch: int,
    global_step: int,
    loss: float,
    throughput: float,
) -> None:
    path = os.path.join(logs_dir, f"{job_id}_status.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": loss,
                    "throughput": throughput,
                },
                f,
            )
        os.replace(tmp, path)
    except Exception:
        pass


class DatasetUnavailableError(RuntimeError):
    """The worker has no copy of the requested dataset."""


def build_optimizer(model: nn.Module, training_args: dict) -> torch.optim.Optimizer:
    """Rebuild the controller's optimizer from its spec ({"type", "kwargs"}), else SGD(lr)."""
    spec = training_args.get("optimizer")
    if not spec:
        return torch.optim.SGD(model.parameters(), lr=training_args.get("lr", 0.01))
    cls = getattr(torch.optim, spec["type"], None)
    if not (isinstance(cls, type) and issubclass(cls, torch.optim.Optimizer)):
        raise ValueError(f"Unknown optimizer 'torch.optim.{spec['type']}'")
    # Drop keys this torch version does not know (controller and Pi torch may differ).
    accepted = inspect.signature(cls).parameters
    kwargs = {
        k: tuple(v) if isinstance(v, list) else v
        for k, v in (spec.get("kwargs") or {}).items()
        if k in accepted and v is not None
    }
    return cls(model.parameters(), **kwargs)


def build_loss(training_args: dict) -> nn.Module:
    """Rebuild the controller's loss from its spec ({"type", "kwargs"}), else CrossEntropyLoss."""
    spec = training_args.get("loss")
    if not spec:
        return nn.CrossEntropyLoss()
    cls = getattr(nn, spec["type"], None)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise ValueError(f"Unknown loss 'torch.nn.{spec['type']}'")
    return cls(**(spec.get("kwargs") or {}))


def run_training():
    parser = argparse.ArgumentParser(description="Optimus Worker Training Process")
    parser.add_argument("--training-args", type=str, default="{}", help="JSON training arguments")
    args = parser.parse_args()

    training_args = json.loads(args.training_args) if args.training_args else {}

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = int(os.environ.get("MASTER_PORT", 29500))
    job_id = os.environ.get("OPTIMUS_JOB_ID", "default_job")
    dataset_id = os.environ.get("OPTIMUS_DATASET_ID", "default_dataset")
    dataset_version = os.environ.get("OPTIMUS_DATASET_VERSION", "v1")
    checkpoint_dir = os.environ.get("OPTIMUS_CHECKPOINT_DIR", "/srv/checkpoints")
    resume_path = os.environ.get("OPTIMUS_RESUME_CHECKPOINT")
    logs_dir = training_args.get("logs_dir", "./logs")

    logger = get_logger(rank)
    logger.info(f"Starting worker training process for job '{job_id}' (World size: {world_size})")

    # Resolve user factories from import paths ("module:function") before joining the
    # process group, so a bad path fails fast. Nothing sent over the network is
    # unpickled or evaluated; we only import modules already present on this worker.
    model_factory_path = training_args.get("model_factory") or model_factory_path_from_name(
        training_args.get("model")
    )
    dataset_factory_path = training_args.get("dataset_factory") or DEFAULT_DATASET_FACTORY
    model_factory = resolve_factory(model_factory_path)
    dataset_factory = resolve_factory(dataset_factory_path)

    # 1. Initialize PyTorch DDP via Gloo
    init_distributed(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend="gloo",
    )

    checkpoint_mgr = CheckpointManager(base_dir=checkpoint_dir, job_id=job_id)

    # 2. Locate this worker's copy of the dataset (NFS mount or local storage).
    # A missing dataset is fatal: silently training on random tensors would report a
    # "successful" job whose model learned nothing. Synthetic data is opt-in only.
    allow_synthetic = bool(training_args.get("allow_synthetic_data", False))
    candidates = [
        f"/mnt/nfs-origin/{dataset_id}/{dataset_version}",
        f"/srv/datasets/{dataset_id}/{dataset_version}",
    ]
    dataset_dir = next((c for c in candidates if os.path.isdir(c)), None)
    if dataset_dir is None and not allow_synthetic:
        raise DatasetUnavailableError(
            f"Dataset '{dataset_id}' version '{dataset_version}' not found on this worker. "
            f"Tried: {', '.join(candidates)}. Mount or copy the dataset there "
            "(see README), or pass allow_synthetic_data=True to train on random data."
        )

    num_classes = training_args.get("num_classes", 10)
    dataset_kwargs = dict(training_args.get("dataset_kwargs") or {})
    if dataset_factory_path == DEFAULT_DATASET_FACTORY:
        uses_mlp = model_factory_path == DEFAULT_MODEL_FACTORY
        dataset_kwargs.setdefault("num_classes", num_classes)
        dataset_kwargs.setdefault(
            "num_samples", training_args.get("num_samples", 1000 if uses_mlp else 100)
        )
        dataset_kwargs.setdefault(
            "input_shape", training_args.get("input_shape", [32] if uses_mlp else [3, 32, 32])
        )

    dataset = dataset_factory(dataset_dir, **dataset_kwargs)
    logger.info(
        f"Dataset from '{dataset_factory_path}' (data_dir={dataset_dir}): {len(dataset)} samples"
    )

    if "num_classes" not in training_args and hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    model_kwargs = dict(training_args.get("model_kwargs") or {})
    if accepts_kwarg(model_factory, "num_classes"):
        model_kwargs.setdefault("num_classes", num_classes)
    model = model_factory(**model_kwargs)
    logger.info(f"Model from '{model_factory_path}' with kwargs {model_kwargs}")

    optimizer = build_optimizer(model, training_args)
    criterion = build_loss(training_args)
    logger.info(
        f"Optimizer: {optimizer.__class__.__name__} {optimizer.defaults}; "
        f"loss: {criterion.__class__.__name__} {(training_args.get('loss') or {}).get('kwargs', {})}"
    )

    # Wrap model with DistributedDataParallel
    ddp_model = nn.parallel.DistributedDataParallel(model)

    start_epoch = 0
    global_step = 0

    # 3. Resume from checkpoint if requested
    if resume_path and os.path.isdir(resume_path):
        meta = checkpoint_mgr.load_checkpoint(
            checkpoint_dir=resume_path,
            model=ddp_model,
            optimizer=optimizer,
        )
        start_epoch = meta.epoch + 1
        global_step = meta.global_step
        logger.info(f"Resumed from checkpoint: starting at epoch {start_epoch}, step {global_step}")

    # 4. Construct dataset and distributed sampler
    batch_size = training_args.get("batch_size", 32)
    epochs = training_args.get("epochs", 5)

    sampler = OptimusDistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # 5. Training loop
    try:
        for epoch in range(start_epoch, epochs):
            sampler.set_epoch(epoch)
            ddp_model.train()
            epoch_loss = 0.0
            epoch_start = time.time()
            samples_processed = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                output = ddp_model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                samples_processed += len(batch_x)
                global_step += 1

            duration = max(0.001, time.time() - epoch_start)
            avg_loss = epoch_loss / max(1, len(loader))
            throughput = samples_processed / duration

            logger.info(
                f"Epoch {epoch}/{epochs-1} complete - Loss: {avg_loss:.4f}, Throughput: {throughput:.1f} samples/s"
            )

            # Update status file for optimusd
            update_status_file(logs_dir, job_id, epoch, global_step, avg_loss, throughput)

            # Rank 0 commits atomic checkpoint
            if rank == 0:
                checkpoint_mgr.save_checkpoint(
                    model=ddp_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    dataset_id=dataset_id,
                    dataset_version=dataset_version,
                    world_size=world_size,
                    loss=avg_loss,
                )

            # Synchronize ranks across epoch boundary
            dist.barrier()

        logger.info(f"Training completed successfully for job '{job_id}'")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    run_training()
