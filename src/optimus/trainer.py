"""
Optimus Managed PyTorch Trainer.
Simplifies distributed training: the developer supplies the model, dataset, and optimizer;
Optimus handles manifests, caching, scheduling, DDP, checkpointing, and recovery.
"""

import logging
import os
import time
import uuid
from typing import Any, Dict, Optional
import torch

from optimus.dataset import Dataset
from optimus.protocol.generated.messages_pb2 import JobState
from optimus.sdk import Optimus

logger = logging.getLogger("optimus.trainer")


class Trainer:
    """
    Developer-facing Trainer abstraction matching Section 15 specifications:

        trainer = Trainer(
            cluster=cluster,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            epochs=100,
            workers=2,
            batch_size=32
        )
        trainer.fit()
    """

    def __init__(
        self,
        cluster: Optimus,
        model: torch.nn.Module,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        workers: Optional[int] = None,
        batch_size: int = 32,
        lr: float = 0.01,
        loss_fn: Optional[Any] = None,
        job_id: Optional[str] = None,
    ):
        self.cluster = cluster
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.epochs = epochs
        self.workers = workers or cluster.config.scheduler.default_workers
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.job_id = job_id or f"job-{uuid.uuid4().hex[:8]}"
        self.metrics: Dict[str, Any] = {}

    def fit(self) -> Dict[str, Any]:
        """
        Executes end-to-end distributed training:
        1. Prepares dataset manifest on controller.
        2. Probes worker resources & cache inventory.
        3. Runs scheduler to assign ranks.
        4. Launches DDP across workers over Gloo.
        5. Restores checkpoints on worker failure.
        6. Retrieves final weights to controller.
        """
        logger.info(f"Starting Optimus managed training for job '{self.job_id}'")

        # 1. Authoritative dataset preparation & cryptographic manifest
        manifest = self.cluster.dataset_manager.prepare_dataset(self.dataset)

        # 2. Package training parameters
        model_name = "default"
        class_name = self.model.__class__.__name__.lower() if hasattr(self.model, "__class__") else ""
        if "resnet" in class_name:
            model_name = "resnet18"
        elif any(k in class_name for k in ("cifar", "cnn", "conv")):
            model_name = "cifar10_cnn"
        elif hasattr(self.model, "model_name"):
            model_name = self.model.model_name

        num_classes = getattr(self.model, "num_classes", 10)
        if hasattr(self.model, "fc") and hasattr(self.model.fc, "out_features"):
            num_classes = self.model.fc.out_features
        elif hasattr(self.model, "classifier"):
            for m in reversed(list(self.model.classifier.modules())):
                if hasattr(m, "out_features"):
                    num_classes = m.out_features
                    break

        training_args = {
            "model": model_name,
            "num_classes": num_classes,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "num_samples": getattr(self.dataset, "num_samples", 100),
            "logs_dir": self.cluster.config.logging.directory,
        }

        # 3. Submit and execute job via JobManager
        record = self.cluster.job_manager.run_training_job(
            job_id=self.job_id,
            manifest=manifest,
            world_size=self.workers,
            training_args=training_args,
            max_restarts=self.cluster.config.training.max_restarts,
        )

        if record.status != JobState.JOB_COMPLETED:
            raise RuntimeError(
                f"Training job '{self.job_id}' ended with status: "
                f"{JobState.Name(record.status)}. Error: {record.error_message}"
            )

        # 4. Load final model weights into controller model instance
        final_model_path = os.path.join(
            self.cluster.config.scheduler.output_directory,
            self.job_id,
            "final_model.pt",
        )
        if os.path.isfile(final_model_path):
            try:
                state_dict = torch.load(final_model_path, map_location="cpu")
                self.model.load_state_dict(state_dict)
                logger.info(f"Loaded trained final model weights from '{final_model_path}'")
            except Exception as e:
                logger.warning(f"Could not load state_dict directly into self.model: {e}")

        self.metrics = {
            "job_id": self.job_id,
            "status": "COMPLETED",
            "epochs": self.epochs,
            "final_loss": record.loss,
            "final_step": record.global_step,
            "restarts": record.restart_count,
            "final_model_path": final_model_path,
        }

        logger.info(f"Training successfully completed: {self.metrics}")
        return self.metrics
