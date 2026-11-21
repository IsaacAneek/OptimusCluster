"""
Tests for Phase 3: Node Discovery using Avahi, Nmap output parsing, and protocol verification.
"""

import socket
import pytest

from optimus.cluster import OptimusConfig, WorkerNodeConfig
from optimus.controller.discovery import NodeDiscoveryManager
from optimus.protocol.generated.messages_pb2 import WorkerState
from optimus.worker.server import WorkerServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_discovery_verify_worker():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-disco-01",
        hostname="localhost",
        port=port,
        cluster_secret="test-secret",
    )
    worker.start()

    config = OptimusConfig()
    config.cluster.secret = "test-secret"
    config.controller.hostname = "localhost"

    disco = NodeDiscoveryManager(config)

    try:
        # Verify legitimate worker responds
        status = disco.verify_worker("127.0.0.1", port)
        assert status is not None
        assert status.worker_id == "pi-disco-01"
        assert status.status == WorkerState.AVAILABLE

        # Verify non-existent port fails gracefully
        assert disco.verify_worker("127.0.0.1", port + 100) is None

    finally:
        worker.stop()


def test_discovery_full_flow():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-disco-02",
        hostname="localhost",
        port=port,
        cluster_secret="secret-abc",
    )
    worker.start()

    config = OptimusConfig()
    config.cluster.secret = "secret-abc"
    config.workers = [
        WorkerNodeConfig(hostname="localhost", port=port, enabled=True),
        WorkerNodeConfig(hostname="unreachable.invalid", port=port, enabled=True),
    ]

    disco = NodeDiscoveryManager(config)

    try:
        nodes = disco.discover()
        assert len(nodes) == 1
        assert nodes[0].worker_id == "pi-disco-02"
        assert nodes[0].hostname == "localhost"
        assert nodes[0].state == WorkerState.AVAILABLE
    finally:
        worker.stop()
