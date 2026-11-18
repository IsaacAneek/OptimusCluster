"""
Example Dataset Generator for Optimus.
Creates a structured sample image/tensor classification dataset with category subfolders.
"""

import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image


def generate_synthetic_dataset(
    output_dir: str = "/srv/datasets/animals/v1",
    num_classes: int = 2,
    images_per_class: int = 50,
    image_size: int = 64,
):
    os.makedirs(output_dir, exist_ok=True)
    classes = ["cat", "dog", "bird", "fish"][:num_classes]

    print(f"Generating synthetic dataset in '{output_dir}'...")
    total_generated = 0

    for cls in classes:
        cls_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        for i in range(images_per_class):
            # Generate random RGB image tensor
            img_tensor = torch.randint(0, 256, (3, image_size, image_size), dtype=torch.uint8)
            img = T.ToPILImage()(img_tensor)

            img_path = os.path.join(cls_dir, f"img_{i:04d}.png")
            img.save(img_path)
            total_generated += 1

    print(f"Successfully generated {total_generated} image files across {len(classes)} classes.")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Optimus Synthetic Dataset Generator")
    parser.add_argument("--output", type=str, default="/srv/datasets/animals/v1", help="Output directory")
    parser.add_argument("--images-per-class", type=int, default=50, help="Images per class")
    args = parser.parse_args()

    generate_synthetic_dataset(output_dir=args.output, images_per_class=args.images_per_class)


if __name__ == "__main__":
    main()
