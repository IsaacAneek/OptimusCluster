"""
CIFAR-10 model and dataset factories for Optimus.

Workers import this module by path ("example.cifar10:build_model"), so it must be
importable on every worker: optimusd runs from the repository root, which puts
the ``example`` package on its import path.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from optimus.training.factories import synthetic_dataset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


class CIFAR10ConvNet(nn.Module):
    """
    4-Layer Convolutional Neural Network designed specifically for 32x32 RGB CIFAR-10
    image classification on edge CPU hardware (Raspberry Pi 5 Cortex-A76).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor: 3 stages of Conv -> BatchNorm -> ReLU -> MaxPool
        self.features = nn.Sequential(
            # Stage 1: 3x32x32 -> 32x16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: 32x16x16 -> 64x8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: 64x8x8 -> 128x4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int = 10) -> nn.Module:
    """Model factory: Trainer(model_factory="example.cifar10:build_model")."""
    return CIFAR10ConvNet(num_classes=num_classes)


def build_dataset(data_dir: Optional[str], synthetic_samples: int = 100) -> Dataset:
    """
    Dataset factory: Trainer(dataset_factory="example.cifar10:build_dataset").
    Loads the CIFAR-10 class folders with CIFAR normalization, or random 3x32x32
    tensors when this worker has no copy of the dataset.
    """
    if data_dir:
        from torchvision import transforms
        from torchvision.datasets import ImageFolder

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        try:
            return ImageFolder(data_dir, transform=transform)
        except (FileNotFoundError, RuntimeError):
            pass  # no class subdirectories yet
    return synthetic_dataset(
        num_samples=synthetic_samples,
        num_classes=len(CIFAR10_CLASSES),
        input_shape=(3, 32, 32),
    )
