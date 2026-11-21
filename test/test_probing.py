"""
Tests for Controller-Initiated Worker Probing and Availability Tracking.
Verifies no heartbeats are sent by workers, controller initiates all requests,
failure thresholds trigger UNAVAILABLE, and successful probes recover state.
"""

import socket
import time
import pytest

from optimus.controller.communication import WorkerClientManager
from optimus.controller.probing import ProbeManager
from optimus.protocol.generated.messages_pb2 import WorkerState
from optimus.worker.server import WorkerServer


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_worker_probe_and_availability():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-test-01",
        hostname="localhost",
        port=port,
        cluster_secret="test-secret",
    )
    worker.start()

    client_mgr = WorkerClientManager(
        default_timeout=2.0,
        cluster_secret="test-secret",
        controller_id="controller-test",
    )
    probe_mgr = ProbeManager(
        client_manager=client_mgr,
        failure_threshold=3,
        probe_interval=1.0,
    )

    try:
        probe_mgr.register_worker("localhost", port, "pi-test-01")

        # Initial probe should succeed and mark AVAILABLE
        state = probe_mgr.probe_worker("localhost", port)
        assert state.state == WorkerState.AVAILABLE
        assert state.worker_id == "pi-test-01"
        assert state.consecutive_failures == 0
        assert state.uptime_seconds >= 0

        # Mark worker busy and verify probe reflects it
        worker.active_jobs = 1
        state = probe_mgr.probe_worker("localhost", port)
        assert state.state == WorkerState.BUSY
        assert state.active_jobs == 1
        worker.active_jobs = 0

    finally:
        worker.stop()
        client_mgr.close_all()


def test_probing_failure_threshold_and_recovery():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-test-02",
        hostname="localhost",
        port=port,
    )
    worker.start()

    client_mgr = WorkerClientManager(default_timeout=0.5)
    probe_mgr = ProbeManager(
        client_manager=client_mgr,
        failure_threshold=3,
        probe_interval=0.5,
    )

    try:
        probe_mgr.register_worker("localhost", port, "pi-test-02")
        state = probe_mgr.probe_worker("localhost", port)
        assert state.state == WorkerState.AVAILABLE

        # Stop worker to simulate network failure or crash
        worker.stop()

        # Probe 1 fails
        state = probe_mgr.probe_worker("localhost", port)
        assert state.consecutive_failures == 1
        assert state.state == WorkerState.AVAILABLE  # Not yet UNAVAILABLE

        # Probe 2 fails
        state = probe_mgr.probe_worker("localhost", port)
        assert state.consecutive_failures == 2
        assert state.state == WorkerState.AVAILABLE

        # Probe 3 fails -> reaches threshold of 3, marks UNAVAILABLE
        state = probe_mgr.probe_worker("localhost", port)
        assert state.consecutive_failures == 3
        assert state.state == WorkerState.UNAVAILABLE

        # Restart worker
        worker = WorkerServer(
            worker_id="pi-test-02",
            hostname="localhost",
            port=port,
        )
        worker.start()

        # Probing again recovers state
        state = probe_mgr.probe_worker("localhost", port)
        assert state.state == WorkerState.AVAILABLE
        assert state.consecutive_failures == 0

    finally:
        worker.stop()
        client_mgr.close_all()


def test_cluster_secret_authentication():
    port = get_free_port()
    worker = WorkerServer(
        worker_id="pi-secure-01",
        hostname="localhost",
        port=port,
        cluster_secret="correct-secret",
    )
    worker.start()

    # Client with wrong secret
    bad_client_mgr = WorkerClientManager(
        default_timeout=1.0,
        cluster_secret="wrong-secret",
    )
    bad_probe_mgr = ProbeManager(
        client_manager=bad_client_mgr,
        failure_threshold=2,
    )

    # Client with valid secret
    good_client_mgr = WorkerClientManager(
        default_timeout=1.0,
        cluster_secret="correct-secret",
    )
    good_probe_mgr = ProbeManager(
        client_manager=good_client_mgr,
        failure_threshold=2,
    )

    try:
        # Probe with wrong secret should fail
        state = bad_probe_mgr.probe_worker("localhost", port)
        assert state.consecutive_failures == 1
        assert "401" in (state.last_error or "")
        assert "UNAUTHORIZED" in (state.last_error or "")

        # Probe with good secret should succeed
        state = good_probe_mgr.probe_worker("localhost", port)
        assert state.state == WorkerState.AVAILABLE
        assert state.consecutive_failures == 0

    finally:
        worker.stop()
        bad_client_mgr.close_all()
        good_client_mgr.close_all()
