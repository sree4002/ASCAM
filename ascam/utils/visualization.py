"""Visualization utilities for detection results."""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 10,
    confidences: Optional[List[float]] = None,
    show_conf: bool = False
) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image (BGR format)
        boxes: List of boxes as (x1, y1, x2, y2)
        color: Box color in BGR format
        thickness: Line thickness
        confidences: Optional confidence scores for each box
        show_conf: Whether to show confidence scores

    Returns:
        Image with boxes drawn
    """
    result = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        # Optionally show confidence score
        if show_conf and confidences and i < len(confidences):
            conf_text = f"{confidences[i]:.2f}"
            font_scale = 1.0
            font_thick = 2
            (tw, th), _ = cv2.getTextSize(
                conf_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                font_thick
            )
            cv2.putText(
                result,
                conf_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_thick,
                lineType=cv2.LINE_AA
            )

    return result


def add_count_label(
    image: np.ndarray,
    count: int,
    label_text: str = "Total swellings:",
    position: str = "top-right",
    color: Tuple[int, int, int] = (0, 0, 255),
    font_scale: float = 3.0,
    thickness: int = 8,
    margin: int = 10
) -> np.ndarray:
    """
    Add swelling count label to image.

    Args:
        image: Input image
        count: Number to display
        label_text: Text label
        position: Label position ('top-right', 'top-left', 'bottom-right', 'bottom-left')
        color: Text color in BGR
        font_scale: Font size scale
        thickness: Text thickness
        margin: Margin from edges

    Returns:
        Image with label added
    """
    result = image.copy()
    text = f"{label_text} {count}"

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness
    )

    # Calculate position
    h, w = image.shape[:2]
    if position == "top-right":
        x = w - text_width - margin
        y = margin + text_height
    elif position == "top-left":
        x = margin
        y = margin + text_height
    elif position == "bottom-right":
        x = w - text_width - margin
        y = h - margin
    elif position == "bottom-left":
        x = margin
        y = h - margin
    else:
        x = w - text_width - margin
        y = margin + text_height

    # Draw text
    cv2.putText(
        result,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA
    )

    return result


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    labels: Tuple[str, str] = ("Original", "Detected")
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated images.

    Args:
        original: Original image
        annotated: Annotated image
        labels: Labels for original and annotated images

    Returns:
        Combined image
    """
    # Ensure same height
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]

    if h1 != h2:
        # Resize to match height
        target_h = max(h1, h2)
        if h1 < target_h:
            original = cv2.resize(original, (int(w1 * target_h / h1), target_h))
        if h2 < target_h:
            annotated = cv2.resize(annotated, (int(w2 * target_h / h2), target_h))

    # Concatenate horizontally
    combined = np.hstack([original, annotated])

    return combined
