"""
ASCAM - Axonal Swelling Classification and Analysis Model

A deep learning pipeline for automated detection and quantification
of axonal swellings in histological brain sections.
"""

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
