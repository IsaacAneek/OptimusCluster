"""
Optimus CIFAR-10 Distributed CNN Training Example.
Demonstrates training a custom 4-layer Convolutional Neural Network (ConvNet)
on the CIFAR-10 10-class image classification dataset across Raspberry Pi 5 workers
using the Optimus SDK and PyTorch DDP over Gloo.

Automatically checks for and downloads CIFAR-10 if not present locally.
"""

import argparse
import os
import sys
import tarfile
import urllib.request
import torch
import torch.optim as optim

# Allow both `python example/train_cifar10.py` and `python -m example.train_cifar10`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from example.cifar10 import CIFAR10_CLASSES, build_model  # noqa: E402
from optimus import Dataset, Optimus, Trainer  # noqa: E402

FASTAI_CIFAR10_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"


def ensure_cifar10_dataset(
    output_dir: str = "/srv/datasets/cifar10/v1",
    samples_per_class: int = 100,
) -> str:
    """
    Ensures CIFAR-10 dataset is downloaded and structured into class subdirectories.
    If not already populated, streams the archive from the FastAI S3 mirror.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check if dataset already has images
    existing_classes = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d)) and d in CIFAR10_CLASSES
    ]
    if len(existing_classes) == len(CIFAR10_CLASSES):
        total_existing = sum(
            len(os.listdir(os.path.join(output_dir, c)))
            for c in existing_classes
        )
        if total_existing >= len(CIFAR10_CLASSES) * min(samples_per_class, 50):
            print(f"CIFAR-10 dataset already prepared in '{output_dir}' ({total_existing} images).")
            return output_dir

    print(f"Downloading CIFAR-10 dataset from mirror to '{output_dir}' ({samples_per_class} samples/class)...")
    counts = {c: 0 for c in CIFAR10_CLASSES}

    req = urllib.request.Request(
        FASTAI_CIFAR10_URL,
        headers={"User-Agent": "OptimusDL/1.0 (Distributed ML Platform)"}
    )

    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            with tarfile.open(fileobj=resp, mode="r|gz") as tar:
                for member in tar:
                    if not member.isfile() or "train/" not in member.name:
                        continue
                    parts = member.name.split("/")
                    if len(parts) >= 4:
                        cls_name = parts[2]
                        filename = parts[3]
                        if cls_name in counts and counts[cls_name] < samples_per_class:
                            cls_dir = os.path.join(output_dir, cls_name)
                            os.makedirs(cls_dir, exist_ok=True)
                            f_out = os.path.join(cls_dir, filename)
                            extracted = tar.extractfile(member)
                            if extracted:
                                with open(f_out, "wb") as f:
                                    f.write(extracted.read())
                                counts[cls_name] += 1

                    if all(counts[c] >= samples_per_class for c in CIFAR10_CLASSES):
                        break
        print(f"Successfully downloaded {sum(counts.values())} CIFAR-10 images across {len(CIFAR10_CLASSES)} classes.")
    except Exception as e:
        # Never fall back to generated images here: they would land in the real dataset
        # folder and the job would "succeed" on noise.
        raise SystemExit(
            f"Could not download CIFAR-10 into '{output_dir}': {e}\n"
            "Check network access, or place the class folders there manually, then re-run."
        ) from e

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Optimus CIFAR-10 Distributed CNN Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-worker mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--workers", type=int, default=2, help="Number of DDP worker nodes")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Samples to download per class")
    args = parser.parse_args()

    # 1. Ensure CIFAR-10 dataset is present on the controller's authoritative storage
    dataset_path = "/srv/datasets/cifar10/v1"
    ensure_cifar10_dataset(output_dir=dataset_path, samples_per_class=args.samples_per_class)

    # 2. Connect to the Optimus cluster controller
    print(f"\nConnecting to Optimus Cluster using '{args.config}'...")
    cluster = Optimus(config=args.config)

    # Display live worker telemetry
    print("\n--- Live Cluster Worker Telemetry ---")
    print(cluster.probe())
    print("--------------------------------------\n")

    # 3. Define the dataset abstraction
    dataset = Dataset(
        source=dataset_path,
        dataset_id="cifar10",
        version="v1",
    )

    # 4. Construct the custom CIFAR-10 CNN model and optimizer
    model = build_model(num_classes=10)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Architecture: CIFAR10ConvNet ({total_params:,} trainable parameters)")

    # 5. Initialize the managed Trainer
    # Workers rebuild the model and dataset by importing these factories; nothing in
    # src/optimus needs to know about CIFAR-10.
    trainer = Trainer(
        cluster=cluster,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        model_factory="example.cifar10:build_model",
        dataset_factory="example.cifar10:build_dataset",
        model_kwargs={"num_classes": 10},
        epochs=args.epochs,
        workers=args.workers,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    # 6. Execute managed distributed training across the cluster
    print(f"Starting distributed training across {args.workers} workers for {args.epochs} epochs...")
    metrics = trainer.fit()

    print("\n--- CIFAR-10 Training Job Completed Successfully ---")
    print(f"Job ID: {metrics['job_id']}")
    print(f"Final Epoch: {metrics['epochs']}")
    print(f"Final Loss: {metrics['final_loss']:.4f}")
    print(f"Total Restarts: {metrics['restarts']}")
    print(f"Model Artifact: {metrics['final_model_path']}")
    print("----------------------------------------------------\n")

    # 7. Verify and load trained weights into local model instance
    if os.path.isfile(metrics["final_model_path"]):
        state_dict = torch.load(metrics["final_model_path"], map_location="cpu")
        model.load_state_dict(state_dict)
        print("Trained CIFAR10ConvNet weights successfully verified and loaded on the controller!")


if __name__ == "__main__":
    main()
