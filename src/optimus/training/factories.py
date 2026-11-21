"""
Optimus factory resolution.

Workers never receive pickled objects or source code from the network. Instead, the
controller sends *import paths* of the form ``"package.module:function"`` and each
worker imports and calls them locally. A factory path is therefore only usable if the
module is importable on every worker (installed in the worker's environment, or
present under optimusd's working directory / ``PYTHONPATH``).

Contracts:

* model factory:   ``factory(**model_kwargs) -> torch.nn.Module``.
  If the factory accepts ``num_classes``, Optimus fills it in unless the caller set it.
* dataset factory: ``factory(data_dir, **dataset_kwargs) -> torch.utils.data.Dataset``.
  ``data_dir`` is the dataset directory found on the worker (NFS mount or local copy),
  or ``None`` when no copy of the dataset exists on that worker.
"""

import importlib
import inspect
import os
import re
from typing import Any, Callable, Optional, Sequence

import logging
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset

logger = logging.getLogger("optimus.training.factories")

# "pkg.module:attr" or "pkg.module:Outer.attr" -- identifiers only, nothing to evaluate.
_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"
_FACTORY_PATH_RE = re.compile(rf"^{_IDENT}(\.{_IDENT})*:{_IDENT}(\.{_IDENT})*$")

DEFAULT_MODEL_FACTORY = "optimus.training.factories:mlp"
DEFAULT_DATASET_FACTORY = "optimus.training.factories:default_dataset"

# Short names accepted in the legacy ``"model"`` training argument.
LEGACY_MODEL_ALIASES = {
    "default": DEFAULT_MODEL_FACTORY,
    "mlp": DEFAULT_MODEL_FACTORY,
    "resnet18": "torchvision.models:resnet18",
}
_REMOVED_MODEL_ALIASES = ("cifar10_cnn", "cifar_cnn", "convnet", "cnn")


class FactoryResolutionError(ValueError):
    """Raised when a factory path is malformed or cannot be imported."""


def is_factory_path(value: Any) -> bool:
    return isinstance(value, str) and bool(_FACTORY_PATH_RE.match(value))


def resolve_factory(path: str) -> Callable[..., Any]:
    """Imports ``"module:attr"`` and returns the callable it names."""
    if not is_factory_path(path):
        raise FactoryResolutionError(
            f"Invalid factory path {path!r}: expected 'package.module:function'"
        )
    module_name, attr_path = path.split(":", 1)
    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as e:
        raise FactoryResolutionError(
            f"Cannot import module {module_name!r} for factory {path!r}: {e}. "
            "The module must be importable on every worker (installed, or under the "
            "optimusd's working directory / PYTHONPATH)."
        ) from e
    for part in attr_path.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise FactoryResolutionError(
                f"Module {module_name!r} has no attribute {attr_path!r} (factory {path!r})"
            ) from e
    if not callable(obj):
        raise FactoryResolutionError(f"Factory {path!r} resolved to a non-callable {type(obj).__name__}")
    return obj


def accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """True if ``fn`` takes a keyword argument called ``name`` (or ``**kwargs``)."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == name and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
        for p in params
    )


def model_factory_path_from_name(name: Optional[str]) -> str:
    """Maps the legacy ``"model"`` training argument to a factory path."""
    name = name or "default"
    if is_factory_path(name):
        return name
    if name in LEGACY_MODEL_ALIASES:
        return LEGACY_MODEL_ALIASES[name]
    if name in _REMOVED_MODEL_ALIASES:
        raise FactoryResolutionError(
            f"Built-in model {name!r} was moved out of Optimus. Pass "
            "model_factory='example.cifar10:build_model' (or your own 'module:function')."
        )
    raise FactoryResolutionError(
        f"Unknown model {name!r}. Use one of {sorted(LEGACY_MODEL_ALIASES)} or a "
        "'module:function' factory path."
    )


def factory_path_of(fn: Callable[..., Any]) -> str:
    """Returns the ``"module:qualname"`` import path of a top-level function or class."""
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        raise FactoryResolutionError(f"Cannot derive an import path for {fn!r}")
    if module == "__main__":
        raise FactoryResolutionError(
            f"{qualname!r} is defined in the script being run (__main__), which workers "
            "cannot import. Move it into an importable module and pass it from there, or "
            "pass its 'module:function' path as a string."
        )
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise FactoryResolutionError(
            f"{qualname!r} is a lambda or nested function; factories must be module-level."
        )
    path = f"{module}:{qualname}"
    if not is_factory_path(path):
        raise FactoryResolutionError(f"Derived factory path {path!r} is not a valid import path")
    return path


# ---------------------------------------------------------------------------
# Built-in generic factories (no task-specific models live in Optimus itself)
# ---------------------------------------------------------------------------

def mlp(num_classes: int = 10, input_dim: int = 32, hidden_dim: int = 64) -> nn.Module:
    """Small MLP used when no model factory is given (smoke tests, demos)."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )


def synthetic_dataset(
    data_dir: Optional[str] = None,
    num_samples: int = 100,
    num_classes: int = 10,
    input_shape: Sequence[int] = (3, 32, 32),
) -> Dataset:
    """Random tensors with random labels; ignores ``data_dir``."""
    x = torch.randn(num_samples, *input_shape)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def _has_class_subdirs(data_dir: Optional[str]) -> bool:
    if not data_dir or not os.path.isdir(data_dir):
        return False
    return any(
        os.path.isdir(os.path.join(data_dir, d)) and not d.startswith((".", "tmp"))
        for d in os.listdir(data_dir)
    )


def image_folder_dataset(
    data_dir: str,
    image_size: int = 32,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> Dataset:
    """torchvision ``ImageFolder`` over ``data_dir`` resized to ``image_size``."""
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    steps = [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    if mean is not None and std is not None:
        steps.append(transforms.Normalize(tuple(mean), tuple(std)))
    return ImageFolder(data_dir, transform=transforms.Compose(steps))


def default_dataset(
    data_dir: Optional[str] = None,
    num_samples: int = 100,
    num_classes: int = 10,
    input_shape: Sequence[int] = (3, 32, 32),
    image_size: int = 32,
) -> Dataset:
    """ImageFolder over ``data_dir``; synthetic tensors only when ``data_dir`` is None
    (the worker passes None only if the job opted in with ``allow_synthetic_data``)."""
    if data_dir is not None:
        if not _has_class_subdirs(data_dir):
            raise FileNotFoundError(f"No class subdirectories found in dataset dir '{data_dir}'")
        return image_folder_dataset(data_dir, image_size=image_size)
    return synthetic_dataset(
        num_samples=num_samples, num_classes=num_classes, input_shape=input_shape
    )
