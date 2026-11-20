"""
Optimus Worker Training Process Entrypoint.
Standard execution script spawned by the worker daemon to run DDP training.
Initializes Gloo process group, runs epoch batches, synchronizes gradients,
and allows Rank 0 to write atomic checkpoints.
"""

import argparse
import json
import logging
import os
import sys
import time
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from optimus.controller.checkpoint import CheckpointManager
from optimus.training.ddp import cleanup_distributed, init_distributed
from optimus.training.sampler import OptimusDistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Rank %(rank)s] %(message)s",
)


def get_logger(rank: int):
    logger = logging.getLogger("optimus.worker_entry")
    extra = {"rank": str(rank)}
    return logging.LoggerAdapter(logger, extra)


class CIFAR10CNN(nn.Module):
    """4-layer Convolutional Neural Network tailored for 32x32 CIFAR-10 image classification on CPU."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


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

    # 1. Initialize PyTorch DDP via Gloo
    init_distributed(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        backend="gloo",
    )

    checkpoint_mgr = CheckpointManager(base_dir=checkpoint_dir, job_id=job_id)

    # 2. Check for real dataset on NFS/local storage
    dataset = None
    dataset_dir = None
    for cand in [
        f"/mnt/nfs-origin/{dataset_id}/{dataset_version}",
        f"/srv/datasets/{dataset_id}/{dataset_version}",
    ]:
        if os.path.isdir(cand):
            dataset_dir = cand
            break

    if dataset_dir:
        subdirs = [
            d for d in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, d))
            and not d.startswith(".") and not d.startswith("tmp")
        ]
        if subdirs:
            try:
                from torchvision import transforms
                from torchvision.datasets import ImageFolder
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
                dataset = ImageFolder(dataset_dir, transform=transform)
                logger.info(
                    f"Loaded ImageFolder dataset from '{dataset_dir}' with {len(dataset)} samples across {len(dataset.classes)} classes"
                )
            except Exception as e:
                logger.warning(f"Could not load ImageFolder from '{dataset_dir}': {e}. Falling back to synthetic tensors")
                dataset = None

    # Setup model and optimizer
    model_type = training_args.get("model", "default")
    num_classes = training_args.get("num_classes", len(dataset.classes) if dataset and hasattr(dataset, "classes") else 10)

    if model_type == "resnet18":
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        if dataset is None:
            x_data = torch.randn(training_args.get("num_samples", 100), 3, 32, 32)
            y_data = torch.randint(0, num_classes, (training_args.get("num_samples", 100),))
            dataset = TensorDataset(x_data, y_data)
    elif model_type in ("cifar10_cnn", "cifar_cnn", "convnet", "cnn"):
        model = CIFAR10CNN(num_classes=num_classes)
        if dataset is None:
            x_data = torch.randn(training_args.get("num_samples", 100), 3, 32, 32)
            y_data = torch.randint(0, num_classes, (training_args.get("num_samples", 100),))
            dataset = TensorDataset(x_data, y_data)
    else:
        hidden_dim = training_args.get("hidden_dim", 64)
        model = nn.Sequential(
            nn.Linear(32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        if dataset is None:
            x_data = torch.randn(training_args.get("num_samples", 1000), 32)
            y_data = torch.randint(0, num_classes, (training_args.get("num_samples", 1000),))
            dataset = TensorDataset(x_data, y_data)

    optimizer = torch.optim.SGD(model.parameters(), lr=training_args.get("lr", 0.01))
    criterion = nn.CrossEntropyLoss()

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

            # Update status file for worker daemon
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
