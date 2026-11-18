"""
Optimus Cluster Scheduler.
Implements the unified locality-aware weighted scheduling algorithm used by both
the Python SDK and the Web Dashboard. Generates selected_nodes.json atomically.
"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from optimus.controller.cache_registry import ClusterCacheCoordinator
from optimus.controller.probing import ProbeManager, WorkerNodeState
from optimus.dataset import DatasetManifest
from optimus.protocol.generated.messages_pb2 import WorkerState

logger = logging.getLogger("optimus.controller.scheduler")


@dataclass
class SelectedNode:
    hostname: str
    port: int
    worker_id: str
    rank: int
    cached_bytes: int
    required_bytes: int
    cache_locality: float
    score: float


@dataclass
class ScheduleResult:
    job_id: str
    dataset_id: str
    dataset_version: str
    world_size: int
    output_file: str
    selected_nodes: List[SelectedNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "world_size": self.world_size,
            "selected_nodes": [asdict(n) for n in self.selected_nodes],
        }


class Scheduler:
    """Unified scheduler matching Section 14 specifications."""

    def __init__(
        self,
        probe_manager: ProbeManager,
        cache_coordinator: ClusterCacheCoordinator,
        runs_dir: str = "./runs",
    ):
        self.probe_manager = probe_manager
        self.cache_coordinator = cache_coordinator
        self.runs_dir = os.path.abspath(runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)

    def schedule(
        self,
        job_id: str,
        manifest: DatasetManifest,
        world_size: int = 2,
        required_ram_mb: int = 500,
        refresh_probes: bool = True,
    ) -> ScheduleResult:
        """
        Executes node filtering, weighted scoring, rank assignment, and atomic output generation.
        """
        # Step 1: Probe fresh resources from all workers if requested
        if refresh_probes:
            self.probe_manager.probe_all_resources()
        states = self.probe_manager.get_all_states()

        # Step 2: Filter eligible workers
        eligible: List[Tuple[WorkerNodeState, float, int]] = []
        total_req_bytes = manifest.total_bytes

        for s in states:
            if s.state != WorkerState.AVAILABLE:
                logger.debug(f"Excluding {s.hostname}: state={WorkerState.Name(s.state)}")
                continue

            if s.active_jobs > 0:
                logger.debug(f"Excluding {s.hostname}: active_jobs={s.active_jobs}")
                continue

            if s.ram_available_mb < required_ram_mb:
                logger.debug(
                    f"Excluding {s.hostname}: insufficient RAM ({s.ram_available_mb} MB < {required_ram_mb} MB)"
                )
                continue

            locality, cached_bytes = self.cache_coordinator.compute_locality(
                s.hostname, s.port, manifest
            )
            missing_bytes = max(0, total_req_bytes - cached_bytes)
            available_disk_bytes = s.cache_disk_available_gb * (1024**3)

            if available_disk_bytes < missing_bytes:
                logger.debug(
                    f"Excluding {s.hostname}: insufficient disk space for missing files "
                    f"({available_disk_bytes} bytes < {missing_bytes} bytes missing)"
                )
                continue

            eligible.append((s, locality, cached_bytes))

        if len(eligible) < world_size:
            raise RuntimeError(
                f"Scheduler could not find enough eligible workers: requested {world_size}, found {len(eligible)}"
            )

        # Step 3: Compute normalized scores
        max_ram = max((e[0].ram_available_mb for e in eligible), default=1)
        max_disk = max((e[0].cache_disk_available_gb for e in eligible), default=1)

        scored_nodes: List[Tuple[WorkerNodeState, float, int, float]] = []

        for s, locality, cached_bytes in eligible:
            norm_ram = min(1.0, s.ram_available_mb / float(max(1, max_ram)))
            norm_cpu = max(0.0, 1.0 - (s.cpu_percent / 100.0))
            norm_disk = min(1.0, s.cache_disk_available_gb / float(max(1, max_disk)))

            # Thermal health: 35C is ideal (1.0), 85C is throttle point (0.0)
            if s.temperature_available and s.temperature_c > 0:
                norm_thermal = max(0.0, min(1.0, 1.0 - (max(0.0, s.temperature_c - 35.0) / 50.0)))
                # Standard weights
                score = (
                    0.50 * locality
                    + 0.20 * norm_ram
                    + 0.15 * norm_cpu
                    + 0.10 * norm_disk
                    + 0.05 * norm_thermal
                )
            else:
                # Renormalize remaining weights when thermal is unavailable
                # sum of remaining weights: 0.50 + 0.20 + 0.15 + 0.10 = 0.95
                score = (
                    (0.50 / 0.95) * locality
                    + (0.20 / 0.95) * norm_ram
                    + (0.15 / 0.95) * norm_cpu
                    + (0.10 / 0.95) * norm_disk
                )

            scored_nodes.append((s, locality, cached_bytes, score))

        # Step 4: Rank workers by score descending
        scored_nodes.sort(key=lambda x: x[3], reverse=True)
        chosen = scored_nodes[:world_size]

        selected: List[SelectedNode] = []
        for rank, (s, locality, cached_bytes, score) in enumerate(chosen):
            selected.append(
                SelectedNode(
                    hostname=s.hostname,
                    port=s.port,
                    worker_id=s.worker_id,
                    rank=rank,
                    cached_bytes=cached_bytes,
                    required_bytes=total_req_bytes,
                    cache_locality=round(locality, 4),
                    score=round(score, 4),
                )
            )

        # Step 5: Atomically write selected_nodes.json
        job_dir = os.path.join(self.runs_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)
        output_file = os.path.join(job_dir, "selected_nodes.json")

        result = ScheduleResult(
            job_id=job_id,
            dataset_id=manifest.dataset_id,
            dataset_version=manifest.version,
            world_size=world_size,
            output_file=output_file,
            selected_nodes=selected,
        )

        temp_out = f"{output_file}.tmp.{os.getpid()}"
        with open(temp_out, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_out, output_file)

        logger.info(f"Scheduler assigned {world_size} nodes for job '{job_id}' -> '{output_file}'")
        return result
