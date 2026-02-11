"""Image loading and preprocessing utilities."""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load an image from file path.

    Args:
        image_path: Path to the image file

    Returns:
        Image as numpy array in BGR format, or None if loading fails
    """
    try:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def preprocess_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (200, 200),
    normalize: bool = True,
    to_rgb: bool = True
) -> np.ndarray:
    """
    Preprocess image for model inference.

    Args:
        image: Input image in BGR format
        target_size: Target (height, width) for resizing
        normalize: Whether to normalize pixel values to [0, 1]
        to_rgb: Whether to convert from BGR to RGB

    Returns:
        Preprocessed image
    """
    # Resize
    processed = cv2.resize(image, (target_size[1], target_size[0]))

    # Convert to RGB if requested
    if to_rgb:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    # Normalize
    if normalize:
        processed = processed.astype(np.float32) / 255.0

    return processed


def save_image(image: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    Save image to file.

    Args:
        image: Image array to save
        output_path: Destination path

    Returns:
        True if successful, False otherwise
    """
    try:
        success = cv2.imwrite(str(output_path), image)
        if success:
            logger.info(f"Saved image to {output_path}")
        else:
            logger.error(f"Failed to save image to {output_path}")
        return success
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def get_image_files(directory: Union[str, Path]) -> list:
    """
    Get all image files from a directory using case-insensitive extension matching.

    Args:
        directory: Directory to search

    Returns:
        List of Path objects for image files
    """
    directory = Path(directory)
    if not directory.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []

    supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    return sorted(image_files)
