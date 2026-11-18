"""
Tests for Phase 8: PyTorch Distributed Training and Executor Lifecycle.
Verifies Gloo DDP initialization, DistributedSampler partitioning,
TrainingExecutor submission and monitoring, and graceful process teardown.
"""

import socket
import subprocess
import time
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from optimus.protocol.generated.messages_pb2 import (
    JobResponse,
    JobState,
    JobSubmit,
)
from optimus.training.ddp import cleanup_distributed, init_distributed
from optimus.training.sampler import OptimusDistributedSampler
from optimus.worker.executor import TrainingExecutor


def get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def test_distributed_sampler_partitioning():
    x = torch.randn(100, 4)
    y = torch.randint(0, 2, (100,))
    ds = TensorDataset(x, y)

    sampler_rank0 = OptimusDistributedSampler(ds, num_replicas=2, rank=0, shuffle=True, seed=42)
    sampler_rank1 = OptimusDistributedSampler(ds, num_replicas=2, rank=1, shuffle=True, seed=42)

    indices_0 = list(sampler_rank0)
    indices_1 = list(sampler_rank1)

    assert len(indices_0) == 50
    assert len(indices_1) == 50

    # Ensure mutually exclusive partitioning (no duplicates across ranks)
    assert set(indices_0).isdisjoint(set(indices_1))
    assert set(indices_0).union(set(indices_1)) == set(range(100))


def test_single_rank_ddp_gloo_init():
    port = get_free_port()
    try:
        rank, world_size = init_distributed(
            rank=0,
            world_size=1,
            master_addr="127.0.0.1",
            master_port=port,
            backend="gloo",
        )
        assert rank == 0
        assert world_size == 1

        # Check torch.distributed is initialized
        assert torch.distributed.is_initialized()
        t = torch.tensor([42.0])
        torch.distributed.all_reduce(t)
        assert t.item() == 42.0
    finally:
        cleanup_distributed()


def test_training_executor_submit_and_stop(tmp_path):
    logs_dir = str(tmp_path / "logs")
    executor = TrainingExecutor(logs_dir=logs_dir)

    req = JobSubmit(
        job_id="test_exec_01",
        dataset_id="test_ds",
        dataset_version="v1",
        rank=0,
        world_size=1,
        master_addr="127.0.0.1",
        master_port=get_free_port(),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        required_ram_mb=100,
    )

    resp = executor.submit_job(req)
    assert resp.accepted is True

    # Immediate status check should report RUNNING
    time.sleep(0.5)
    status_resp = executor.get_status("test_exec_01", "worker-01")
    assert status_resp.status in (JobState.JOB_RUNNING, JobState.JOB_COMPLETED)

    # Stop the job
    executor.stop_job("test_exec_01", reason="Test teardown")
    time.sleep(0.2)
    status_stopped = executor.get_status("test_exec_01", "worker-01")
    assert status_stopped.status == JobState.JOB_STOPPED
