"""
ASCAM YOLO Training
===================
Train YOLOv8s for swelling detection with paper-compliant parameters.

Paper parameters:
- Model: yolov8s.pt
- imgsz: 1280
- Epochs: 500
- Mosaic: 0.0 (disabled for histology)
- Optimizer: AdamW (lr=0.001, lrf=0.01)
- conf=0.25, iou=0.50
- Augmentation: degrees=15, hsv tuned for DAB-stained tissue
"""

import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: str,
    output_dir: str = "models/",
    epochs: int = 500,
    batch_size: int = 16,
    model_name: str = "yolov8s.pt",
    device: str = None,
):
    """
    Train YOLOv8s with paper-compliant parameters.

    Args:
        data_yaml: Path to dataset.yaml (standard Ultralytics format)
        output_dir: Output directory for trained model
        epochs: Number of training epochs
        batch_size: Batch size
        model_name: YOLO model name (default: yolov8s.pt)
        device: Device to use
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    logger.info("=" * 60)
    logger.info("ASCAM YOLO TRAINING PIPELINE")
    logger.info("=" * 60)

    # Load pretrained model
    model = YOLO(model_name)

    # Paper-compliant training arguments
    train_args = {
        "data": str(data_yaml),
        "epochs": epochs,
        "batch": batch_size,
        "imgsz": 1280,
        "patience": 50,
        # Optimization
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 3,
        # Paper thresholds
        "conf": 0.25,
        "iou": 0.50,
        # Augmentation - mosaic/mixup/copy_paste disabled for histology
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 15,
        "translate": 0.1,
        "scale": 0.3,
        "hsv_h": 0.01,
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        # Output
        "project": str(output_dir / "yolo_training"),
        "name": "train",
        "exist_ok": True,
        # Other
        "workers": 8,
        "seed": 42,
        "verbose": True,
        "plots": True,
        "device": device,
    }

    logger.info("Training Configuration:")
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  imgsz:      1280")
    logger.info(f"  Epochs:     {epochs}")
    logger.info(f"  Batch:      {batch_size}")
    logger.info(f"  Conf:       0.25")
    logger.info(f"  IoU:        0.50")
    logger.info(f"  Mosaic:     OFF (histology)")
    logger.info(f"  Degrees:    15")
    logger.info(f"  Device:     {device}")

    # Train
    results = model.train(**train_args)

    # Copy best model to output directory
    best_model_path = output_dir / "yolo_training" / "train" / "weights" / "best.pt"
    if best_model_path.exists():
        shutil.copy2(best_model_path, output_dir / "yolov8s_best.pt")
        logger.info(f"Best model saved: {output_dir / 'yolov8s_best.pt'}")

    # Validate on test set
    best_model = YOLO(str(best_model_path)) if best_model_path.exists() else model
    test_results = best_model.val(
        data=str(data_yaml), split="test", imgsz=1280, conf=0.25, iou=0.50, verbose=True
    )

    metrics = {
        "mAP50": float(test_results.box.map50),
        "mAP50-95": float(test_results.box.map),
        "precision": float(test_results.box.mp),
        "recall": float(test_results.box.mr),
        "conf_threshold": 0.25,
        "iou_threshold": 0.50,
    }

    logger.info(f"\nTest Results:")
    logger.info(f"  mAP@50:     {metrics['mAP50']*100:.2f}%")
    logger.info(f"  mAP@50-95:  {metrics['mAP50-95']*100:.2f}%")
    logger.info(f"  Precision:  {metrics['precision']*100:.2f}%")
    logger.info(f"  Recall:     {metrics['recall']*100:.2f}%")

    # Save results
    with open(output_dir / "yolo_training_results.json", "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": {
                    "model": model_name,
                    "imgsz": 1280,
                    "epochs": epochs,
                    "conf": 0.25,
                    "iou": 0.50,
                    "mosaic": 0.0,
                    "degrees": 15,
                    "scale": 0.3,
                    "hsv_h": 0.01,
                    "hsv_s": 0.5,
                    "hsv_v": 0.3,
                },
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )

    logger.info(f"Training complete! Results saved to {output_dir}")
    return metrics
