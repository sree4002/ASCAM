#!/usr/bin/env python3
"""
ASCAM YOLOv8s Training Script
==============================
Standalone training script for the YOLOv8s swelling detection model.
This is the script used to train the final production model (yolov8s_best.pt).

Training pipeline:
  1. Offline augmentation with albumentations (CLAHE, flips, rotation, etc.)
  2. YOLOv8s training at imgsz=1280 with histology-tuned augmentation
  3. Evaluation with and without TTA (test-time augmentation)

Usage:
  python scripts/ASCAM_YOLO_Training.py --data /path/to/detection --output /path/to/output
  python scripts/ASCAM_YOLO_Training.py --data data/detection --output models/ --epochs 300
"""

import argparse
import os
import shutil
import random

import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8s for axonal swelling detection"
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to detection data directory (must contain train/, val/, test/ with images/ and labels/)"
    )
    parser.add_argument(
        "--output", default="models/",
        help="Output directory for trained model (default: models/)"
    )
    parser.add_argument(
        "--work-dir", default=None,
        help="Working directory for augmented data and training runs (default: <output>/yolo_training_work)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs (default: 300)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--aug-per-image", type=int, default=5,
        help="Number of augmented copies per training image (default: 5)"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use (default: auto-detect GPU, fallback to CPU)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


# ── Augmentation Pipeline ────────────────────────────────────────────────────

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.RandomScale(scale_limit=0.15, p=0.3),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


def augment_dataset(src_images, src_labels, dst_images, dst_labels, augmentations_per_image=5):
    """Apply offline augmentation to training images with bbox-safe transforms."""
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)
    image_files = [f for f in os.listdir(src_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_created = 0

    for img_file in tqdm(image_files, desc="Augmenting"):
        img_path = os.path.join(src_images, img_file)
        image = cv2.imread(img_path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(src_labels, label_file)
        bboxes, class_labels = [], []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_labels.append(int(parts[0]))
                        bboxes.append([float(x) for x in parts[1:5]])

        base_name = os.path.splitext(img_file)[0]
        ext = os.path.splitext(img_file)[1]

        # Copy original
        cv2.imwrite(os.path.join(dst_images, img_file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(dst_labels, label_file))
        total_created += 1

        # Create augmented versions
        for i in range(augmentations_per_image):
            try:
                transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                aug_img_name = f"{base_name}_aug{i}{ext}"
                aug_lbl_name = f"{base_name}_aug{i}.txt"

                cv2.imwrite(os.path.join(dst_images, aug_img_name), cv2.cvtColor(transformed['image'], cv2.COLOR_RGB2BGR))
                with open(os.path.join(dst_labels, aug_lbl_name), 'w') as f:
                    for cls, bbox in zip(transformed['class_labels'], transformed['bboxes']):
                        f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                total_created += 1
            except (ValueError, cv2.error) as e:
                print(f"  ⚠ Skipped augmentation {i} for {img_file}: {e}")
                continue
    return total_created


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Paths
    data_path = os.path.abspath(args.data)
    output_dir = os.path.abspath(args.output)
    work_dir = os.path.abspath(args.work_dir) if args.work_dir else os.path.join(output_dir, "yolo_training_work")
    aug_path = os.path.join(work_dir, "augmented_data")

    # Device
    if args.device is not None:
        device = args.device
    else:
        device = 0 if torch.cuda.is_available() else 'cpu'

    os.makedirs(output_dir, exist_ok=True)

    # Clear previous work dir if it exists
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
        print("✓ Cleared previous training files")

    os.makedirs(work_dir, exist_ok=True)

    print(f"✓ Data path:    {data_path}")
    print(f"✓ Output dir:   {output_dir}")
    print(f"✓ Working dir:  {work_dir}")
    print(f"✓ Device:       {device}")
    print()

    # ── Step 1: Offline Augmentation ─────────────────────────────────────────

    print("=== Step 1: Creating augmented training dataset ===")
    n_train = augment_dataset(
        f"{data_path}/train/images", f"{data_path}/train/labels",
        f"{aug_path}/train/images", f"{aug_path}/train/labels",
        args.aug_per_image
    )
    print(f"✓ Training images: {n_train}")

    # Copy val/test unchanged
    for split in ['val', 'test']:
        shutil.copytree(f"{data_path}/{split}/images", f"{aug_path}/{split}/images")
        shutil.copytree(f"{data_path}/{split}/labels", f"{aug_path}/{split}/labels")
        print(f"✓ {split}: {len(os.listdir(f'{aug_path}/{split}/images'))} images")

    # Create data.yaml
    with open(f"{aug_path}/data.yaml", 'w') as f:
        f.write(f"path: {aug_path}\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: ['swelling']")
    print("✓ data.yaml created")
    print()

    # ── Step 2: Train YOLOv8s ────────────────────────────────────────────────

    print("=== Step 2: Training YOLOv8s ===")
    model = YOLO('yolov8s.pt')

    model.train(
        data=f"{aug_path}/data.yaml",
        epochs=args.epochs,
        patience=50,
        imgsz=1280,
        batch=args.batch_size,
        device=device,

        # Augmentation settings (histology-tuned)
        degrees=15,
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.5,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,

        # Disabled for histology
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,

        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=5,

        project=f"{work_dir}/runs",
        name="yolov8s_1280_augmented",
        exist_ok=True,
        verbose=True,
        plots=True,
        seed=args.seed,
    )
    print("\n✓ Training complete!")

    # ── Step 3: Save model ───────────────────────────────────────────────────

    print("\n=== Step 3: Saving model ===")
    best_weights = f"{work_dir}/runs/yolov8s_1280_augmented/weights/best.pt"
    dest_path = os.path.join(output_dir, "yolov8s_best.pt")

    shutil.copy(best_weights, dest_path)
    print(f"✓ Model saved to: {dest_path}")

    # Copy full training results
    results_dir = os.path.join(output_dir, "yolo_training_results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    shutil.copytree(f"{work_dir}/runs/yolov8s_1280_augmented", results_dir)
    print(f"✓ Results saved to: {results_dir}/")

    # ── Step 4: Evaluate on Test Set ─────────────────────────────────────────

    print("\n=== Step 4: Evaluation ===")
    best_model = YOLO(dest_path)

    # Without TTA
    results = best_model.val(data=f"{aug_path}/data.yaml", split='test', imgsz=1280, conf=0.25, iou=0.5)
    print("\n" + "=" * 50)
    print("TEST RESULTS (no TTA)")
    print("=" * 50)
    print(f"Precision: {results.box.mp:.1%}")
    print(f"Recall:    {results.box.mr:.1%}")
    print(f"mAP50:     {results.box.map50:.1%}")
    f1 = 2 * results.box.mp * results.box.mr / (results.box.mp + results.box.mr + 1e-6)
    print(f"F1:        {f1:.1%}")

    # With TTA (FINAL MODEL CONFIGURATION)
    results_tta = best_model.val(data=f"{aug_path}/data.yaml", split='test', imgsz=1280, conf=0.25, iou=0.5, augment=True)
    print("\n" + "=" * 50)
    print("TEST RESULTS (with TTA) — FINAL MODEL")
    print("=" * 50)
    print(f"Precision: {results_tta.box.mp:.1%}")
    print(f"Recall:    {results_tta.box.mr:.1%}")
    print(f"mAP50:     {results_tta.box.map50:.1%}")
    f1_tta = 2 * results_tta.box.mp * results_tta.box.mr / (results_tta.box.mp + results_tta.box.mr + 1e-6)
    print(f"F1:        {f1_tta:.1%}")

    print(f"\n✓ Done! Production model: {dest_path}")


if __name__ == "__main__":
    main()
