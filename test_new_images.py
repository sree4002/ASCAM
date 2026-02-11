"""
Quick test script: run trained classifier on new images.

Usage:
    python test_new_images.py /path/to/new/images
    python test_new_images.py /path/to/new/images --show
    python test_new_images.py /path/to/new/images --model models/classifier.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ascam.models.classifier import SwellingClassifier

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def collect_images(folder: Path) -> list[Path]:
    """Gather all image files from the given folder."""
    return sorted(
        p for p in folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def run_predictions(model_path: str, image_dir: str, show: bool = False):
    """Load model, predict on every image in image_dir, print results."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        print(f"Error: '{image_dir}' is not a valid directory.")
        sys.exit(1)

    image_paths = collect_images(image_dir)
    if not image_paths:
        print(f"No images found in '{image_dir}'.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images in {image_dir}\n")

    # Load classifier
    classifier = SwellingClassifier(
        model_path=model_path,
        image_size=(224, 224),
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f"Model loaded from {model_path}")
    print(f"Device: {classifier.device}\n")

    # Run batch prediction
    results = classifier.predict_batch(
        image_paths, return_probs=True, batch_size=32
    )

    # Print results table
    print(f"{'Filename':<40} {'Prediction':<15} {'Confidence':>10}")
    print("-" * 67)

    positive_count = 0
    for path, (has_swelling, confidence) in zip(image_paths, results):
        label = "POSITIVE" if has_swelling else "NEGATIVE"
        if has_swelling:
            positive_count += 1
        print(f"{path.name:<40} {label:<15} {confidence:>9.4f}")

    # Summary
    total = len(image_paths)
    print("-" * 67)
    print(f"Total: {total}  |  Positive: {positive_count}  |  "
          f"Negative: {total - positive_count}")

    # Optionally save and display images with predictions
    if show and image_paths:
        n = len(image_paths)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if n == 1:
            axes = [axes]
        else:
            axes = axes.flat

        for ax, path, (has_swelling, conf) in zip(axes, image_paths, results):
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            label = "POSITIVE" if has_swelling else "NEGATIVE"
            color = "red" if has_swelling else "green"
            ax.set_title(f"{path.name}\n{label} ({conf:.3f})",
                         fontsize=9, color=color)
            ax.axis("off")

        # Hide unused subplots
        for ax in list(axes)[n:]:
            ax.axis("off")

        plt.tight_layout()
        # Save figure to file, then show
        save_name = image_dir.name.replace(" ", "_") + "_predictions.png"
        save_path = Path("outputs") / save_name
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nFigure saved to: {save_path}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test trained ASCAM classifier on new images"
    )
    parser.add_argument(
        "image_dir",
        help="Folder containing new test images"
    )
    parser.add_argument(
        "--model", default="models/classifier.pt",
        help="Path to trained model checkpoint (default: models/classifier.pt)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display images with predictions in a matplotlib grid"
    )
    args = parser.parse_args()

    run_predictions(args.model, args.image_dir, args.show)
