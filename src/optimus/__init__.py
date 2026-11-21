"""
Optimus: Lightweight Distributed ML Training System
"""

from optimus.cluster import OptimusConfig, load_config
from optimus.controller.communication import WorkerClient, WorkerClientManager
from optimus.controller.probing import ProbeManager, WorkerNodeState
from optimus.dataset import Dataset, DatasetManifest
from optimus.sdk import Optimus
from optimus.trainer import Trainer
from optimus.worker.server import WorkerServer

__version__ = "0.1.0"


def __getattr__(name):
    # Imported lazily so ``python -m optimus.worker.optimusd`` runs cleanly.
    if name in ("OptimusDaemon", "WorkerDaemon"):
        from optimus.worker import optimusd

        return getattr(optimusd, name)
    raise AttributeError(f"module 'optimus' has no attribute {name!r}")

__all__ = [
    "Optimus",
    "Dataset",
    "DatasetManifest",
    "Trainer",
    "OptimusConfig",
    "load_config",
    "WorkerClient",
    "WorkerClientManager",
    "ProbeManager",
    "WorkerNodeState",
    "WorkerServer",
    "OptimusDaemon",
    "WorkerDaemon",
]
