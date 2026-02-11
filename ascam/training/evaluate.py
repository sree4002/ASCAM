"""
ASCAM Evaluation
================
Threshold calibration and metrics evaluation.

Usage:
    python -m ascam.training.evaluate \
        --classifier models/classifier.pt \
        --detector models/yolov8s_best.pt \
        --data data/detection/test/ \
        --sweep-conf 0.01 0.50 0.01
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def evaluate_thresholds(
    detector_path: str,
    test_images_dir: str,
    conf_range: tuple = (0.01, 0.50, 0.01),
    iou_threshold: float = 0.50,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Sweep confidence thresholds and report detection counts.

    Args:
        detector_path: Path to YOLO model weights
        test_images_dir: Directory containing test images
        conf_range: (start, stop, step) for confidence sweep
        iou_threshold: Fixed IOU threshold
        output_dir: Directory to save results

    Returns:
        Dictionary with threshold sweep results
    """
    from ultralytics import YOLO

    model = YOLO(detector_path)
    test_dir = Path(test_images_dir)

    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_files = [f for f in test_dir.iterdir()
                   if f.is_file() and f.suffix.lower() in extensions]

    if not image_files:
        logger.error(f"No images found in {test_dir}")
        return {}

    logger.info(f"Evaluating {len(image_files)} images across confidence thresholds")

    start, stop, step = conf_range
    thresholds = np.arange(start, stop + step, step)

    results = []

    for conf in thresholds:
        conf = round(float(conf), 3)
        total_detections = 0
        images_with_detections = 0

        for img_path in image_files:
            preds = model.predict(
                source=str(img_path),
                conf=conf,
                iou=iou_threshold,
                verbose=False
            )[0]

            n_det = len(preds.boxes) if preds.boxes is not None else 0
            total_detections += n_det
            if n_det > 0:
                images_with_detections += 1

        result = {
            'conf_threshold': conf,
            'total_detections': total_detections,
            'images_with_detections': images_with_detections,
            'total_images': len(image_files),
            'mean_detections_per_image': total_detections / len(image_files)
        }
        results.append(result)

        logger.info(
            f"  conf={conf:.3f}: {total_detections} detections "
            f"({images_with_detections}/{len(image_files)} images)"
        )

    sweep_results = {
        'iou_threshold': iou_threshold,
        'image_count': len(image_files),
        'sweep': results
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'threshold_sweep.json', 'w') as f:
            json.dump(sweep_results, f, indent=2)
        logger.info(f"Results saved to {output_dir / 'threshold_sweep.json'}")

    return sweep_results


def evaluate_classifier(
    classifier_path: str,
    test_dir: str,
    output_dir: Optional[str] = None
) -> Dict:
    """
    Evaluate CNN classifier on test data.

    Args:
        classifier_path: Path to CNN model weights
        test_dir: Directory with swelling/ and no_swelling/ subdirectories
        output_dir: Directory to save results

    Returns:
        Dictionary with evaluation metrics
    """
    from ascam.models.classifier import SwellingClassifier

    classifier = SwellingClassifier(model_path=classifier_path)

    test_dir = Path(test_dir)
    true_labels = []
    pred_labels = []
    confidences = []

    for subdir in test_dir.iterdir():
        if not subdir.is_dir():
            continue

        name_lower = subdir.name.lower().strip()
        positive_names = {'swelling', 'swellings', 'positive', 'pos'}
        negative_names = {'no_swelling', 'no swelling', 'no_swellings',
                          'negative', 'neg', 'non_swelling', 'non-swelling'}

        if name_lower in positive_names:
            true_label = 1
        elif name_lower in negative_names:
            true_label = 0
        else:
            continue

        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        for img_path in subdir.iterdir():
            if img_path.suffix.lower() not in extensions:
                continue

            has_swelling, conf = classifier.predict_single(img_path, return_prob=True)
            true_labels.append(true_label)
            pred_labels.append(1 if has_swelling else 0)
            confidences.append(conf)

    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    accuracy = (true_labels == pred_labels).mean()
    tp = ((true_labels == 1) & (pred_labels == 1)).sum()
    fp = ((true_labels == 0) & (pred_labels == 1)).sum()
    fn = ((true_labels == 1) & (pred_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'total_samples': len(true_labels),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }

    logger.info(f"Classifier Evaluation:")
    logger.info(f"  Accuracy:  {accuracy*100:.2f}%")
    logger.info(f"  Precision: {precision*100:.2f}%")
    logger.info(f"  Recall:    {recall*100:.2f}%")
    logger.info(f"  F1 Score:  {f1*100:.2f}%")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'classifier_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)

    return metrics


def main():
    """CLI entry point for evaluation."""
    parser = argparse.ArgumentParser(description='ASCAM Evaluation')
    parser.add_argument('--classifier', help='Path to CNN model weights')
    parser.add_argument('--detector', help='Path to YOLO model weights')
    parser.add_argument('--data', required=True, help='Path to test data directory')
    parser.add_argument('--output', default='./evaluation', help='Output directory')
    parser.add_argument(
        '--sweep-conf', nargs=3, type=float, metavar=('START', 'STOP', 'STEP'),
        help='Confidence threshold sweep range (e.g., 0.01 0.50 0.01)'
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.detector and args.sweep_conf:
        evaluate_thresholds(
            detector_path=args.detector,
            test_images_dir=args.data,
            conf_range=tuple(args.sweep_conf),
            output_dir=args.output
        )

    if args.classifier:
        evaluate_classifier(
            classifier_path=args.classifier,
            test_dir=args.data,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
