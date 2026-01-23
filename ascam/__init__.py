"""
ASCAM - Axonal Swelling Classification and Analysis Model

A deep learning pipeline for automated detection and quantification
of axonal swellings in histological brain sections.
"""

import os

# Fix TensorFlow/PyTorch conflict - set environment variables and import order
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF info messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

# Import PyTorch before TensorFlow to prevent segmentation fault on macOS
import torch  # noqa: E402, F401

__version__ = "1.0.0"
__author__ = "ASCAM Team"

from ascam.models.classifier import SwellingClassifier
from ascam.models.detector import SwellingDetector
from ascam.pipeline import ASCAMPipeline

__all__ = [
    "SwellingClassifier",
    "SwellingDetector",
    "ASCAMPipeline",
]
