"""
Optimus Cluster Configuration Manager.
Parses, validates, and manages cluster-wide configuration settings.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


DISCOVER_HINT = (
    "Run `optimus discover` (or press Scan in the dashboard) to generate the worker list."
)


class ConfigError(ValueError):
    """Configuration is missing or unusable; the message says how to fix it."""


@dataclass
class ClusterMetadata:
    name: str = "optimus"
    protocol_version: int = 1
    secret: str = ""


@dataclass
class ControllerConfig:
    hostname: str = "localhost"
    tcp_port: int = 45870
    dashboard_port: int = 8080


@dataclass
class DiscoveryConfig:
    method: str = "mdns_nmap"
    scan_interval: int = 30
    subnet: str = ""  # empty = auto-detect from the local interface
    use_configured_hosts: bool = True


@dataclass
class WorkerNodeConfig:
    hostname: str
    port: int = 45870
    enabled: bool = True
    worker_id: Optional[str] = None


@dataclass
class CommunicationConfig:
    request_timeout: float = 3.0
    probe_interval: float = 5.0
    failure_threshold: int = 3
    max_message_size: int = 4194304  # 4 MiB


@dataclass
class DatasetConfig:
    controller_storage: str = "/srv/datasets"
    nfs_server: str = ""  # filled in by `optimus discover`; used by deploy/install.sh
    nfs_export: str = "/srv/datasets"
    worker_nfs_mount: str = "/mnt/nfs-origin"
    fuse_mount: str = "/mnt/dataset"


@dataclass
class CacheConfig:
    directory: str = "/var/cache/optimus"
    database: str = "/var/cache/optimus/cache.db"
    max_capacity_gb: int = 50


@dataclass
class TrainingConfig:
    backend: str = "gloo"
    checkpoint_directory: str = "/srv/checkpoints"
    checkpoint_nfs_export: str = "/srv/checkpoints"
    max_restarts: int = 3


@dataclass
class SchedulerConfig:
    output_directory: str = "./runs"
    default_workers: int = 2


@dataclass
class LoggingConfig:
    level: str = "INFO"
    directory: str = "./logs"


@dataclass
class OptimusConfig:
    cluster: ClusterMetadata = field(default_factory=ClusterMetadata)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    workers: List[WorkerNodeConfig] = field(default_factory=list)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimusConfig":
        cluster_data = data.get("cluster", {})
        controller_data = data.get("controller", {})
        discovery_data = data.get("discovery", {})
        workers_data = data.get("workers", [])
        communication_data = data.get("communication", {})
        dataset_data = data.get("dataset", {})
        cache_data = data.get("cache", {})
        training_data = data.get("training", {})
        scheduler_data = data.get("scheduler", {})
        logging_data = data.get("logging", {})

        workers = []
        for w in workers_data:
            if isinstance(w, dict) and "hostname" in w:
                workers.append(
                    WorkerNodeConfig(
                        hostname=w["hostname"],
                        port=w.get("port", controller_data.get("tcp_port", 45870)),
                        enabled=w.get("enabled", True),
                        worker_id=w.get("worker_id"),
                    )
                )

        config = cls(
            cluster=ClusterMetadata(**cluster_data),
            controller=ControllerConfig(**controller_data),
            discovery=DiscoveryConfig(**discovery_data),
            workers=workers,
            communication=CommunicationConfig(**communication_data),
            dataset=DatasetConfig(**dataset_data),
            cache=CacheConfig(**cache_data),
            training=TrainingConfig(**training_data),
            scheduler=SchedulerConfig(**scheduler_data),
            logging=LoggingConfig(**logging_data),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, config_path: str = "config.yaml") -> "OptimusConfig":
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}. {DISCOVER_HINT}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls.from_dict(data)

    def require_workers(self) -> None:
        """Errors clearly when the generated host list is missing or empty."""
        if not any(w.enabled for w in self.workers):
            raise ConfigError(
                "No workers are configured (config 'workers' is missing or empty). "
                + DISCOVER_HINT
            )

    def validate(self) -> None:
        """Validates configuration sanity and thresholds."""
        if not self.controller.hostname:
            raise ValueError("Controller hostname cannot be empty.")
        if not (1024 <= self.controller.tcp_port <= 65535):
            raise ValueError(f"Invalid controller TCP port: {self.controller.tcp_port}")
        if self.communication.request_timeout <= 0:
            raise ValueError("communication.request_timeout must be greater than 0.")
        if self.communication.probe_interval <= 0:
            raise ValueError("communication.probe_interval must be greater than 0.")
        if self.communication.failure_threshold < 1:
            raise ValueError("communication.failure_threshold must be >= 1.")
        if self.communication.max_message_size < 1024:
            raise ValueError("communication.max_message_size must be >= 1024 bytes.")
        if self.scheduler.default_workers < 1:
            raise ValueError("scheduler.default_workers must be >= 1.")


def load_config(path: str = "config.yaml") -> OptimusConfig:
    return OptimusConfig.load(path)
