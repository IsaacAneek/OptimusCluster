"""
Tests for Phase 7: Simple Locality-Aware Scheduler.
Verifies exclusion rules, weighted scoring, cache locality preference,
and atomic generation of selected_nodes.json.
"""

import json
import os
import pytest

from optimus.controller.cache_registry import ClusterCacheCoordinator, WorkerCacheRecord
from optimus.controller.communication import WorkerClientManager
from optimus.controller.probing import ProbeManager, WorkerNodeState
from optimus.controller.scheduler import Scheduler
from optimus.dataset import DatasetManifest, FileEntry
from optimus.protocol.generated.messages_pb2 import WorkerState


class MockCacheCoordinator:
    def __init__(self, localities):
        self.localities = localities

    def compute_locality(self, hostname, port, manifest):
        loc, cached = self.localities.get(hostname, (0.0, 0))
        return loc, cached


def test_scheduler_exclusions_and_scoring(tmp_path):
    client_mgr = WorkerClientManager()
    probe_mgr = ProbeManager(client_mgr)

    # 4 workers with different profiles:
    # Worker 1: unavailable
    probe_mgr._workers[("pi01.local", 45870)] = WorkerNodeState(
        hostname="pi01.local", port=45870, worker_id="pi-01",
        state=WorkerState.UNAVAILABLE, ram_available_mb=4000, cache_disk_available_gb=50
    )
    # Worker 2: low RAM
    probe_mgr._workers[("pi02.local", 45870)] = WorkerNodeState(
        hostname="pi02.local", port=45870, worker_id="pi-02",
        state=WorkerState.AVAILABLE, ram_available_mb=200, cache_disk_available_gb=50 # low RAM
    )
    # Worker 3: healthy, 20% locality
    probe_mgr._workers[("pi03.local", 45870)] = WorkerNodeState(
        hostname="pi03.local", port=45870, worker_id="pi-03",
        state=WorkerState.AVAILABLE, cpu_percent=20.0, ram_available_mb=4000, cache_disk_available_gb=50
    )
    # Worker 4: healthy, 90% locality
    probe_mgr._workers[("pi04.local", 45870)] = WorkerNodeState(
        hostname="pi04.local", port=45870, worker_id="pi-04",
        state=WorkerState.AVAILABLE, cpu_percent=20.0, ram_available_mb=4000, cache_disk_available_gb=50
    )

    manifest = DatasetManifest(
        dataset_id="animals",
        version="v1",
        total_bytes=1000,
        total_files=2,
        files=[FileEntry(path="a.jpg", size=500, sha256="a"), FileEntry(path="b.jpg", size=500, sha256="b")]
    )

    cache_coord = MockCacheCoordinator({
        "pi03.local": (0.20, 200),
        "pi04.local": (0.90, 900),
    })

    scheduler = Scheduler(
        probe_manager=probe_mgr,
        cache_coordinator=cache_coord,
        runs_dir=str(tmp_path / "runs"),
    )

    result = scheduler.schedule(
        job_id="job-test-01",
        manifest=manifest,
        world_size=2,
        required_ram_mb=500,
        refresh_probes=False,
    )
    assert len(result.selected_nodes) == 2

    # High locality node (pi04) must be assigned Rank 0 with higher score
    assert result.selected_nodes[0].hostname == "pi04.local"
    assert result.selected_nodes[0].rank == 0
    assert result.selected_nodes[0].cache_locality == 0.90

    assert result.selected_nodes[1].hostname == "pi03.local"
    assert result.selected_nodes[1].rank == 1

    # Verify atomic output file on disk
    assert os.path.exists(result.output_file)
    with open(result.output_file, "r") as f:
        data = json.load(f)
    assert data["job_id"] == "job-test-01"
    assert len(data["selected_nodes"]) == 2
