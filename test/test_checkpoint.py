"""
Tests for Phase 9: Atomic Checkpointing and Restoration.
Verifies model and optimizer state serialization, latest.json pointer update,
and historical checkpoint pruning.
"""

import json
import os
import pytest
import torch
import torch.nn as nn

from optimus.controller.checkpoint import CheckpointManager


def test_atomic_checkpoint_lifecycle(tmp_path):
    ckpt_dir = str(tmp_path / "checkpoints")
    mgr = CheckpointManager(base_dir=ckpt_dir, job_id="test_job", max_retained=3)

    model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Save 4 checkpoints
    for ep in range(4):
        mgr.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=ep,
            global_step=(ep + 1) * 10,
            dataset_id="test_ds",
            dataset_version="v1",
            world_size=2,
            loss=1.0 / (ep + 1),
        )

    # Verify latest.json points to checkpoint-0003
    latest_path = mgr.get_latest_checkpoint_dir()
    assert latest_path is not None
    assert latest_path.endswith("checkpoint-0003")

    with open(mgr.latest_pointer_file, "r") as f:
        ptr_data = json.load(f)
    assert ptr_data["epoch"] == 3
    assert ptr_data["step"] == 40

    # Verify pruning retained only 3 checkpoints: 0001, 0002, 0003 (0000 pruned)
    retained = [
        d for d in os.listdir(mgr.job_dir)
        if d.startswith("checkpoint-") and not d.endswith(".tmp")
    ]
    assert len(retained) == 3
    assert "checkpoint-0000" not in retained
    assert "checkpoint-0001" in retained
    assert "checkpoint-0003" in retained

    # Verify state restoration into fresh model
    new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.01)

    meta = mgr.load_checkpoint(latest_path, new_model, new_optimizer)
    assert meta.epoch == 3
    assert meta.global_step == 40
    assert meta.dataset_id == "test_ds"

    # Verify weights match exactly
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
