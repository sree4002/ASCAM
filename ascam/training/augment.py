"""
ASCAM Data Augmentation
=======================
Conservative augmentations preserving morphology.
Supports both CNN images and YOLO image+label augmentation.
"""

import random
import logging
from pathlib import Path
from typing import List, Tuple
from itertools import product

import numpy as np
import cv2
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SafeAugmentor:
    """
    Conservative augmentations that preserve morphology.
    No elastic transforms, no perspective warps, no cutouts.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.rotations = [0, 90, 180, 270]
        self.h_flips = [False, True]
        self.brightness_deltas = [-0.15, 0, 0.15]
        self.contrast_factors = [0.85, 1.0, 1.15]

    def get_geometric_transforms(self) -> List[dict]:
        """Get all geometric transform combinations (4 rot x 2 flip = 8)."""
        transforms = []
        for rot in self.rotations:
            for h_flip in self.h_flips:
                transforms.append({
                    'rotation': rot,
                    'h_flip': h_flip,
                    'v_flip': False
                })
        return transforms

    def get_intensity_transforms(self) -> List[dict]:
        """Get intensity transform combinations (3 x 3 = 9)."""
        transforms = []
        for brightness in self.brightness_deltas:
            for contrast in self.contrast_factors:
                transforms.append({
                    'brightness': brightness,
                    'contrast': contrast
                })
        return transforms

    def apply_geometric(self, image: np.ndarray, transform: dict) -> np.ndarray:
        """Apply geometric transforms to image."""
        img = image.copy()

        if transform['rotation'] == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif transform['rotation'] == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif transform['rotation'] == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if transform['h_flip']:
            img = cv2.flip(img, 1)

        if transform.get('v_flip', False):
            img = cv2.flip(img, 0)

        return img

    def apply_intensity(self, image: np.ndarray, transform: dict) -> np.ndarray:
        """Apply intensity transforms to image."""
        img = image.astype(np.float32)

        if transform['brightness'] != 0:
            img = img + (transform['brightness'] * 255)

        if transform['contrast'] != 1.0:
            mean = img.mean()
            img = (img - mean) * transform['contrast'] + mean

        return np.clip(img, 0, 255).astype(np.uint8)


def transform_bbox(
    x_center: float, y_center: float,
    width: float, height: float,
    rotation: int, h_flip: bool, v_flip: bool
) -> Tuple[float, float, float, float]:
    """
    Transform YOLO bbox coordinates based on geometric transforms.
    All coordinates are normalized (0-1).
    """
    if rotation == 90:
        new_x, new_y = 1 - y_center, x_center
        new_w, new_h = height, width
    elif rotation == 180:
        new_x, new_y = 1 - x_center, 1 - y_center
        new_w, new_h = width, height
    elif rotation == 270:
        new_x, new_y = y_center, 1 - x_center
        new_w, new_h = height, width
    else:
        new_x, new_y = x_center, y_center
        new_w, new_h = width, height

    if h_flip:
        new_x = 1 - new_x
    if v_flip:
        new_y = 1 - new_y

    return new_x, new_y, new_w, new_h


def augment_dataset(
    input_dir: str,
    output_dir: str,
    multiplier: int = 10,
    seed: int = 42
) -> dict:
    """
    Augment all images in a directory (CNN data).

    Args:
        input_dir: Input directory with images
        output_dir: Output directory for augmented images
        multiplier: Target augmented images per original
        seed: Random seed

    Returns:
        Summary statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in input_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    logger.info(f"Found {len(image_files)} images, target: ~{multiplier}x")

    augmentor = SafeAugmentor(seed=seed)
    geo_transforms = augmentor.get_geometric_transforms()
    int_transforms = augmentor.get_intensity_transforms()
    all_combinations = list(product(enumerate(geo_transforms), enumerate(int_transforms)))

    total_created = 0

    for img_path in tqdm(image_files, desc="Augmenting"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if len(all_combinations) > multiplier:
            selected = random.sample(all_combinations, multiplier)
        else:
            selected = all_combinations

        for (geo_idx, geo_t), (int_idx, int_t) in selected:
            aug_image = augmentor.apply_geometric(image, geo_t)
            aug_image = augmentor.apply_intensity(aug_image, int_t)

            stem = img_path.stem
            suffix = img_path.suffix
            out_name = f"{stem}_aug_g{geo_idx}_i{int_idx}{suffix}"
            cv2.imwrite(str(output_dir / out_name), cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            total_created += 1

    summary = {
        'original_images': len(image_files),
        'augmented_images': total_created,
        'multiplier_achieved': total_created / len(image_files) if image_files else 0
    }

    logger.info(f"Augmentation complete: {total_created} images created ({summary['multiplier_achieved']:.1f}x)")
    return summary


def augment_yolo_dataset(
    images_dir: str,
    labels_dir: str,
    output_images_dir: str,
    output_labels_dir: str,
    multiplier: int = 10,
    seed: int = 42
) -> dict:
    """
    Augment YOLO dataset (images + labels together).

    Args:
        images_dir: Input images directory
        labels_dir: Input labels directory (YOLO format)
        output_images_dir: Output images directory
        output_labels_dir: Output labels directory
        multiplier: Target augmentations per image
        seed: Random seed

    Returns:
        Summary statistics
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_images_dir = Path(output_images_dir)
    output_labels_dir = Path(output_labels_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    pairs = []
    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() in extensions:
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                pairs.append((img_path, label_path))

    logger.info(f"Found {len(pairs)} image-label pairs")

    augmentor = SafeAugmentor(seed=seed)
    geo_transforms = augmentor.get_geometric_transforms()
    int_transforms = augmentor.get_intensity_transforms()
    all_combinations = list(product(enumerate(geo_transforms), enumerate(int_transforms)))

    total_created = 0

    for img_path, label_path in tqdm(pairs, desc="Augmenting YOLO data"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(label_path) as f:
            labels = [line.strip().split() for line in f.readlines() if line.strip()]

        if len(all_combinations) > multiplier:
            selected = random.sample(all_combinations, multiplier)
        else:
            selected = all_combinations

        for (geo_idx, geo_t), (int_idx, int_t) in selected:
            aug_image = augmentor.apply_geometric(image, geo_t)
            aug_image = augmentor.apply_intensity(aug_image, int_t)

            aug_labels = []
            for label in labels:
                cls_id = label[0]
                x_center, y_center, bw, bh = map(float, label[1:5])
                new_x, new_y, new_w, new_h = transform_bbox(
                    x_center, y_center, bw, bh,
                    geo_t['rotation'], geo_t['h_flip'], geo_t.get('v_flip', False)
                )
                aug_labels.append(f"{cls_id} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}")

            out_name = f"{img_path.stem}_aug_g{geo_idx}_i{int_idx}"
            cv2.imwrite(
                str(output_images_dir / f"{out_name}{img_path.suffix}"),
                cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
            )
            with open(output_labels_dir / f"{out_name}.txt", 'w') as f:
                f.write('\n'.join(aug_labels))
            total_created += 1

    summary = {
        'original_pairs': len(pairs),
        'augmented_pairs': total_created,
        'multiplier_achieved': total_created / len(pairs) if pairs else 0
    }

    logger.info(f"YOLO augmentation complete: {total_created} pairs created")
    return summary
