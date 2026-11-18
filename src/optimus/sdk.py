"""
Optimus SDK Entrypoint.
Provides the primary developer-facing interface to interact with the Optimus cluster.
"""

import logging
import os
from typing import Optional

from optimus.cluster import OptimusConfig
from optimus.controller.cache_registry import ClusterCacheCoordinator
from optimus.controller.communication import WorkerClientManager
from optimus.controller.dataset_manager import DatasetManager
from optimus.controller.discovery import NodeDiscoveryManager
from optimus.controller.job_manager import JobManager
from optimus.controller.launcher import DDPLauncher
from optimus.controller.probing import ProbeManager
from optimus.controller.scheduler import ScheduleResult, Scheduler

logger = logging.getLogger("optimus.sdk")


class Optimus:
    """
    Optimus Cluster Controller SDK.
    Coordinates worker discovery, status/resource probing, dataset distribution,
    caching registries, scheduling, and distributed training execution.
    """

    def __init__(self, config: str = "config.yaml"):
        self.config_path = config
        self.config = OptimusConfig.load(config)

        # 1. TCP Client Manager
        self.client_manager = WorkerClientManager(
            default_timeout=self.config.communication.request_timeout,
            cluster_secret=self.config.cluster.secret,
            controller_id=self.config.controller.hostname,
            max_message_size=self.config.communication.max_message_size,
        )

        # 2. Probing Manager
        self.probe_manager = ProbeManager(
            client_manager=self.client_manager,
            failure_threshold=self.config.communication.failure_threshold,
            probe_interval=self.config.communication.probe_interval,
        )

        # 3. Discovery Manager
        self.discovery_manager = NodeDiscoveryManager(config=self.config)

        # 4. Cache Coordinator
        self.cache_coordinator = ClusterCacheCoordinator(
            client_manager=self.client_manager
        )

        # 5. Dataset Manager
        self.dataset_manager = DatasetManager(
            storage_root=self.config.dataset.controller_storage
        )

        # 6. Scheduler
        self.scheduler = Scheduler(
            probe_manager=self.probe_manager,
            cache_coordinator=self.cache_coordinator,
            runs_dir=self.config.scheduler.output_directory,
        )

        # 7. DDP Launcher & Job Manager
        self.launcher = DDPLauncher(client_manager=self.client_manager)
        self.job_manager = JobManager(
            scheduler=self.scheduler,
            launcher=self.launcher,
            probe_manager=self.probe_manager,
            client_manager=self.client_manager,
            checkpoint_base_dir=self.config.training.checkpoint_directory,
            runs_dir=self.config.scheduler.output_directory,
        )

        # Register configured worker nodes
        self._register_configured_workers()

    def _register_configured_workers(self) -> None:
        for w in self.config.workers:
            if w.enabled:
                self.probe_manager.register_worker(
                    hostname=w.hostname,
                    port=w.port,
                    worker_id=w.worker_id,
                )

    def discover_nodes(self) -> None:
        """Runs Avahi mDNS resolution and protocol checks to discover live workers."""
        discovered = self.discovery_manager.discover()
        for node in discovered:
            self.probe_manager.register_worker(
                hostname=node.hostname,
                port=node.port,
                worker_id=node.worker_id,
            )

    def schedule(
        self,
        dataset_id: str,
        dataset_version: str = "v1",
        workers: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> ScheduleResult:
        """
        Public scheduler interface invoked via SDK or dashboard.
        """
        manifest = self.dataset_manager.load_manifest(dataset_id, dataset_version)
        if not manifest:
            raise FileNotFoundError(
                f"Manifest for dataset '{dataset_id}:{dataset_version}' not found. "
                "Ensure dataset is prepared before scheduling."
            )

        target_workers = workers or self.config.scheduler.default_workers
        effective_job_id = job_id or f"sched_{dataset_id}_{dataset_version}"

        return self.scheduler.schedule(
            job_id=effective_job_id,
            manifest=manifest,
            world_size=target_workers,
        )

    def probe(self) -> str:
        """Executes fresh status/resource probes and returns the formatted worker table."""
        self.probe_manager.probe_all_resources()
        return self.probe_manager.format_worker_table()

    def close(self) -> None:
        """Closes all network connections and stops probing loops."""
        self.probe_manager.stop_probing_loop()
        self.client_manager.close_all()
