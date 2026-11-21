"""
Tests for import-path model/dataset factories: resolution, Trainer argument packaging,
and an end-to-end single-rank worker run using the CIFAR-10 example factories.
"""

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch.nn as nn

from optimus.trainer import Trainer
from optimus.training import factories
from optimus.training.factories import (
    FactoryResolutionError,
    accepts_kwarg,
    factory_path_of,
    model_factory_path_from_name,
    resolve_factory,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _stub_cluster(tmp_path):
    return SimpleNamespace(
        config=SimpleNamespace(
            scheduler=SimpleNamespace(default_workers=2),
            logging=SimpleNamespace(directory=str(tmp_path / "logs")),
        )
    )


# --- resolution -------------------------------------------------------------

def test_resolve_valid_paths():
    assert resolve_factory("optimus.training.factories:mlp") is factories.mlp
    assert callable(resolve_factory("torchvision.models:resnet18"))


@pytest.mark.parametrize(
    "path",
    [
        "os.system('id')",
        "__import__('os'):system",
        "optimus.training.factories",
        "optimus.training.factories:mlp()",
        "optimus..factories:mlp",
        "",
    ],
)
def test_resolve_rejects_non_import_paths(path):
    with pytest.raises(FactoryResolutionError):
        resolve_factory(path)


def test_resolve_missing_module_and_attr():
    with pytest.raises(FactoryResolutionError, match="Cannot import"):
        resolve_factory("no_such_module_xyz:build")
    with pytest.raises(FactoryResolutionError, match="no attribute"):
        resolve_factory("optimus.training.factories:does_not_exist")


def test_resolve_rejects_non_callable():
    with pytest.raises(FactoryResolutionError, match="non-callable"):
        resolve_factory("optimus.training.factories:DEFAULT_MODEL_FACTORY")


def test_legacy_aliases():
    assert model_factory_path_from_name(None) == factories.DEFAULT_MODEL_FACTORY
    assert model_factory_path_from_name("default") == factories.DEFAULT_MODEL_FACTORY
    assert model_factory_path_from_name("resnet18") == "torchvision.models:resnet18"
    assert model_factory_path_from_name("pkg.mod:fn") == "pkg.mod:fn"
    with pytest.raises(FactoryResolutionError, match="example.cifar10:build_model"):
        model_factory_path_from_name("cifar10_cnn")
    with pytest.raises(FactoryResolutionError, match="Unknown model"):
        model_factory_path_from_name("vgg")


def test_factory_path_of():
    from example.cifar10 import build_model

    assert factory_path_of(build_model) == "example.cifar10:build_model"
    with pytest.raises(FactoryResolutionError, match="lambda or nested"):
        factory_path_of(lambda: None)


def test_accepts_kwarg():
    assert accepts_kwarg(factories.mlp, "num_classes")
    assert accepts_kwarg(lambda **kw: None, "num_classes")
    assert not accepts_kwarg(lambda hidden=3: None, "num_classes")


# --- Trainer ----------------------------------------------------------------

def test_trainer_packages_factory_paths(tmp_path):
    from example.cifar10 import build_model

    trainer = Trainer(
        cluster=_stub_cluster(tmp_path),
        dataset=SimpleNamespace(num_samples=40),
        model_factory=build_model,  # callable -> converted to import path
        dataset_factory="example.cifar10:build_dataset",
        model_kwargs={"num_classes": 7},
        dataset_kwargs={"synthetic_samples": 12},
        workers=1,
    )
    args = trainer.build_training_args()
    json.dumps(args)  # must be JSON-serializable
    assert args["model_factory"] == "example.cifar10:build_model"
    assert args["dataset_factory"] == "example.cifar10:build_dataset"
    assert args["model_kwargs"] == {"num_classes": 7}
    assert args["dataset_kwargs"] == {"synthetic_samples": 12}
    assert args["num_classes"] == 7
    assert "model" not in args
    # Model is built locally from the factory when not supplied.
    assert trainer.model.__class__.__name__ == "CIFAR10ConvNet"
    assert trainer.model.num_classes == 7


def test_trainer_rejects_unidentifiable_model(tmp_path):
    with pytest.raises(ValueError, match="model_factory"):
        Trainer(cluster=_stub_cluster(tmp_path), model=nn.Linear(2, 2), dataset=object(), workers=1)


def test_trainer_rejects_non_json_kwargs(tmp_path):
    with pytest.raises(ValueError, match="JSON-serializable"):
        Trainer(
            cluster=_stub_cluster(tmp_path),
            dataset=object(),
            model_factory="optimus.training.factories:mlp",
            model_kwargs={"layer": nn.Linear(2, 2)},
            workers=1,
        )


def test_trainer_legacy_resnet_instance_warns(tmp_path):
    from torchvision.models import resnet18

    with pytest.warns(DeprecationWarning):
        trainer = Trainer(
            cluster=_stub_cluster(tmp_path),
            model=resnet18(weights=None, num_classes=3),
            dataset=object(),
            workers=1,
        )
    args = trainer.build_training_args()
    assert args["model_factory"] == "torchvision.models:resnet18"
    assert args["num_classes"] == 3


# --- end-to-end worker process ----------------------------------------------

def _run_worker(tmp_path, training_args):
    env = os.environ.copy()
    env.update(
        RANK="0",
        WORLD_SIZE="1",
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(_free_port()),
        OPTIMUS_JOB_ID="factory-e2e",
        OPTIMUS_DATASET_ID="optimus-test-no-such-dataset",
        OPTIMUS_DATASET_VERSION="v0",
        OPTIMUS_CHECKPOINT_DIR=str(tmp_path / "ckpt"),
        GLOO_SOCKET_IFNAME="lo",
    )
    training_args = {"logs_dir": str(tmp_path / "logs"), **training_args}
    os.makedirs(training_args["logs_dir"], exist_ok=True)
    return subprocess.run(
        [sys.executable, "-m", "optimus.training.worker_entry", "--training-args", json.dumps(training_args)],
        cwd=REPO_ROOT,  # optimusd runs from the repo root
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_worker_trains_with_example_factories(tmp_path):
    proc = _run_worker(
        tmp_path,
        {
            "model_factory": "example.cifar10:build_model",
            "dataset_factory": "example.cifar10:build_dataset",
            "model_kwargs": {"num_classes": 10},
            "dataset_kwargs": {"synthetic_samples": 16},
            "epochs": 1,
            "batch_size": 8,
        },
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "example.cifar10:build_model" in out
    assert "Training completed successfully" in out
    with open(tmp_path / "logs" / "factory-e2e_status.json") as f:
        assert json.load(f)["global_step"] == 2


def test_worker_legacy_default_model(tmp_path):
    proc = _run_worker(tmp_path, {"model": "default", "num_samples": 32, "epochs": 1})
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_worker_bad_factory_fails_fast(tmp_path):
    proc = _run_worker(tmp_path, {"model_factory": "no_such_module_xyz:build", "epochs": 1})
    assert proc.returncode != 0
    assert "Cannot import module 'no_such_module_xyz'" in proc.stdout + proc.stderr
