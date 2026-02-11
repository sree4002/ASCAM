"""ASCAM Training Module.

Provides training pipelines for both the CNN classifier and YOLO detector,
data augmentation, and evaluation utilities.
"""

from ascam.training.train_cnn import Trainer, create_model
from ascam.training.evaluate import evaluate_thresholds

__all__ = ["Trainer", "create_model", "evaluate_thresholds"]
