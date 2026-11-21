"""
Optimus Managed PyTorch Trainer.
Simplifies distributed training: the developer supplies the model, dataset, and optimizer;
Optimus handles manifests, caching, scheduling, DDP, checkpointing, and recovery.
"""

import json
import logging
import os
import uuid
import warnings
from typing import Any, Callable, Dict, Optional, Union
import torch

from optimus.dataset import Dataset
from optimus.protocol.generated.messages_pb2 import JobState
from optimus.sdk import Optimus
from optimus.training.factories import (
    FactoryResolutionError,
    accepts_kwarg,
    factory_path_of,
    is_factory_path,
    model_factory_path_from_name,
    resolve_factory,
)

logger = logging.getLogger("optimus.trainer")


FactoryArg = Union[str, Callable[..., Any], None]


class Trainer:
    """
    Developer-facing managed Trainer:

        trainer = Trainer(
            cluster=cluster,
            dataset=dataset,
            model_factory="example.cifar10:build_model",      # importable on workers
            dataset_factory="example.cifar10:build_dataset",  # optional
            model_kwargs={"num_classes": 10},
            epochs=100,
            workers=2,
            batch_size=32,
        )
        trainer.fit()

    Workers rebuild the model and dataset by importing the given factories
    (``"module:function"`` strings, or module-level callables that Optimus turns into
    such strings). See ``optimus.training.factories`` for the factory contracts.

    ``model`` is optional: when given, the trained weights are loaded into it after
    ``fit()``; when omitted, it is built locally from ``model_factory``.
    """

    def __init__(
        self,
        cluster: Optimus,
        model: Optional[torch.nn.Module] = None,
        dataset: Optional[Dataset] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 10,
        workers: Optional[int] = None,
        batch_size: int = 32,
        lr: float = 0.01,
        loss_fn: Optional[Any] = None,
        job_id: Optional[str] = None,
        *,
        model_factory: FactoryArg = None,
        dataset_factory: FactoryArg = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        dataset_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if dataset is None:
            raise ValueError("Trainer requires a dataset")
        self.cluster = cluster
        self.dataset = dataset
        self.optimizer = optimizer
        self.epochs = epochs
        self.workers = workers or cluster.config.scheduler.default_workers
        self.batch_size = batch_size
        self.lr = lr
        self.loss_fn = loss_fn
        self.job_id = job_id or f"job-{uuid.uuid4().hex[:8]}"
        self.metrics: Dict[str, Any] = {}

        self.model_kwargs = dict(model_kwargs or {})
        self.dataset_kwargs = dict(dataset_kwargs or {})
        for name, kw in (("model_kwargs", self.model_kwargs), ("dataset_kwargs", self.dataset_kwargs)):
            try:
                json.dumps(kw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{name} must be JSON-serializable (sent to workers): {e}") from e

        self.model_factory = self._to_factory_path(model_factory)
        self.dataset_factory = self._to_factory_path(dataset_factory)

        if self.model_factory is None:
            if model is None:
                raise ValueError("Trainer requires either model_factory or model")
            self.model_factory = self._legacy_model_factory(model)
        # Fail fast on the controller if the path does not import here either.
        factory_fn = resolve_factory(self.model_factory)
        if self.dataset_factory is not None:
            resolve_factory(self.dataset_factory)

        if model is None:
            kwargs = dict(self.model_kwargs)
            if accepts_kwarg(factory_fn, "num_classes"):
                kwargs.setdefault("num_classes", 10)
            model = factory_fn(**kwargs)
        self.model = model

    @staticmethod
    def _to_factory_path(factory: FactoryArg) -> Optional[str]:
        if factory is None:
            return None
        if isinstance(factory, str):
            if not is_factory_path(factory):
                raise FactoryResolutionError(
                    f"Invalid factory path {factory!r}: expected 'package.module:function'"
                )
            return factory
        if callable(factory):
            return factory_path_of(factory)
        raise TypeError(f"Factory must be a 'module:function' string or a callable, got {type(factory).__name__}")

    @staticmethod
    def _legacy_model_factory(model: torch.nn.Module) -> str:
        """Backward compatibility for callers that pass only a model instance."""
        explicit = getattr(model, "model_name", None)
        if explicit:
            return model_factory_path_from_name(explicit)
        try:
            from torchvision.models import ResNet
            if isinstance(model, ResNet):
                warnings.warn(
                    "Inferring the worker model from a ResNet instance is deprecated; pass "
                    "model_factory='torchvision.models:resnet18' (or your own factory).",
                    DeprecationWarning,
                    stacklevel=3,
                )
                return model_factory_path_from_name("resnet18")
        except ImportError:
            pass
        raise ValueError(
            f"Cannot tell workers how to build {type(model).__name__}. Pass "
            "model_factory='your.module:build_model' (a module-level function importable "
            "on every worker that returns the model)."
        )

    def _infer_num_classes(self) -> int:
        model = self.model
        if hasattr(model, "fc") and hasattr(model.fc, "out_features"):
            return model.fc.out_features
        if hasattr(model, "classifier"):
            for m in reversed(list(model.classifier.modules())):
                if hasattr(m, "out_features"):
                    return m.out_features
        return getattr(model, "num_classes", 10)

    def build_training_args(self) -> Dict[str, Any]:
        """JSON-serializable arguments sent to every worker's training process."""
        args: Dict[str, Any] = {
            "model_factory": self.model_factory,
            "model_kwargs": self.model_kwargs,
            "num_classes": self.model_kwargs.get("num_classes", self._infer_num_classes()),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "num_samples": getattr(self.dataset, "num_samples", 100),
            "logs_dir": self.cluster.config.logging.directory,
        }
        if self.dataset_factory is not None:
            args["dataset_factory"] = self.dataset_factory
            args["dataset_kwargs"] = self.dataset_kwargs
        return args

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

        # 2. Package training parameters (import paths + JSON kwargs only)
        training_args = self.build_training_args()

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
