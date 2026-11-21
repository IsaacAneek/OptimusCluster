"""
Optimus ResNet Distributed Training Example.
Demonstrates the developer experience using the Optimus SDK on the controller machine.
Trains ResNet-18 across Raspberry Pi 5 workers using PyTorch DDP (Gloo backend)
and full-file FUSE caching over NFS.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18

from optimus import Dataset, Optimus, Trainer


def build_resnet_model(num_classes: int = 2) -> nn.Module:
    """Builds a ResNet-18 model configured for CPU training."""
    model = resnet18(weights=None)
    # Modify fc for target number of classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    # 1. Initialize cluster controller configuration
    config_path = os.environ.get("OPTIMUS_CONFIG", "config.yaml")
    print(f"Connecting to Optimus Cluster using '{config_path}'...")
    cluster = Optimus(config=config_path)

    # Display cluster worker status
    print("\n--- Worker Health and Availability ---")
    print(cluster.probe())
    print("--------------------------------------\n")

    # 2. Define dataset (stored on controller, exported via NFS, cached via FUSE)
    dataset = Dataset(
        source="/srv/datasets/animals/v1",
        dataset_id="animals",
        version="v1",
    )

    # 3. Define model and optimizer
    model = build_resnet_model(num_classes=2)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 4. Create managed Trainer
    trainer = Trainer(
        cluster=cluster,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        model_factory="torchvision.models:resnet18",
        model_kwargs={"num_classes": 2},
        epochs=3,
        workers=2,
        batch_size=16,
    )

    # 5. Execute managed distributed training
    print("Starting distributed training pipeline...")
    metrics = trainer.fit()

    print("\n--- Training Job Completed Successfully ---")
    print(f"Job ID: {metrics['job_id']}")
    print(f"Final Epoch: {metrics['epochs']}")
    print(f"Final Loss: {metrics['final_loss']}")
    print(f"Model Artifact: {metrics['final_model_path']}")
    print("-------------------------------------------\n")

    # Developer can directly load final model weights locally on the controller:
    if os.path.isfile(metrics["final_model_path"]):
        model.load_state_dict(torch.load(metrics["final_model_path"], map_location="cpu"))
        print("Final model state successfully loaded on controller!")


if __name__ == "__main__":
    main()
