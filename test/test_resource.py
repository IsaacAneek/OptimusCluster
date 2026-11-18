"""
Tests for Phase 2: Worker Resource Probing.
Verifies CPU, RAM, cache disk space, thermal reading, and resource probing via TCP.
"""

import socket
import pytest

from optimus.controller.communication import WorkerClientManager
from optimus.controller.probing import ProbeManager
from optimus.protocol.generated.messages_pb2 import WorkerState
from optimus.worker.resource_info import ResourceCollector
from optimus.worker.server import WorkerServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_resource_collector_local():
    collector = ResourceCollector(cache_dir="/tmp")
    cpu = collector.get_cpu_percent()
    assert isinstance(cpu, float)
    assert 0.0 <= cpu <= 100.0

    ram_total, ram_avail = collector.get_ram_mb()
    assert ram_total > 0
    assert ram_avail > 0
    assert ram_avail <= ram_total

    disk_total, disk_avail = collector.get_cache_disk_gb()
    assert disk_total > 0
    assert disk_avail >= 0

    temp_c, temp_avail = collector.get_cpu_temperature()
    assert isinstance(temp_avail, bool)
    if temp_avail:
        assert temp_c > 0.0


def test_resource_probing_over_tcp():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-res-01",
        hostname="localhost",
        port=port,
        cache_dir="/tmp",
    )
    worker.start()

    client_mgr = WorkerClientManager(default_timeout=2.0)
    probe_mgr = ProbeManager(client_manager=client_mgr)
    probe_mgr.register_worker("localhost", port, "pi-res-01")

    try:
        state = probe_mgr.probe_worker_resources("localhost", port)
        assert state.state == WorkerState.AVAILABLE
        assert state.worker_id == "pi-res-01"
        assert state.ram_total_mb > 0
        assert state.ram_available_mb > 0
        assert state.cache_disk_total_gb > 0
        assert state.consecutive_failures == 0

        # Verify formatted table contains resource output
        table = probe_mgr.format_worker_table()
        assert "pi-res-01" in table
        assert "RAM (MB)" in table
        assert "AVAILABLE" in table

    finally:
        worker.stop()
        client_mgr.close_all()
