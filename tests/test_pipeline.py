"""Tests for the ASCAMPipeline."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ascam.models.detector import DetectionResult


class TestASCAMPipeline:
    """Tests for ASCAMPipeline with mocked models."""

    @patch('ascam.pipeline.SwellingDetector')
    @patch('ascam.pipeline.SwellingClassifier')
    def test_pipeline_creation(self, mock_classifier_cls, mock_detector_cls):
        mock_classifier_cls.return_value = MagicMock()
        mock_detector_cls.return_value = MagicMock()

        from ascam.pipeline import ASCAMPipeline
        pipeline = ASCAMPipeline(
            classifier_path="model.pt",
            detector_path="yolov8s_best.pt",
            detector_conf=0.25,
            detector_iou=0.50
        )
        assert pipeline is not None

    @patch('ascam.pipeline.SwellingDetector')
    @patch('ascam.pipeline.SwellingClassifier')
    def test_pipeline_skip_classification(self, mock_classifier_cls, mock_detector_cls):
        mock_detector_cls.return_value = MagicMock()

        from ascam.pipeline import ASCAMPipeline
        pipeline = ASCAMPipeline(
            classifier_path="model.pt",
            detector_path="yolov8s_best.pt",
            skip_classification=True
        )
        assert pipeline.classifier is None

    @patch('ascam.pipeline.SwellingDetector')
    @patch('ascam.pipeline.SwellingClassifier')
    def test_get_statistics_empty(self, mock_classifier_cls, mock_detector_cls):
        mock_classifier_cls.return_value = MagicMock()
        mock_detector_cls.return_value = MagicMock()

        from ascam.pipeline import ASCAMPipeline
        pipeline = ASCAMPipeline(
            classifier_path="model.pt",
            detector_path="yolov8s_best.pt"
        )
        stats = pipeline.get_statistics([])
        assert stats == {}

    @patch('ascam.pipeline.SwellingDetector')
    @patch('ascam.pipeline.SwellingClassifier')
    def test_get_statistics(self, mock_classifier_cls, mock_detector_cls):
        mock_classifier_cls.return_value = MagicMock()
        mock_detector_cls.return_value = MagicMock()

        from ascam.pipeline import ASCAMPipeline
        pipeline = ASCAMPipeline(
            classifier_path="model.pt",
            detector_path="yolov8s_best.pt"
        )

        results = [
            DetectionResult(Path("a.jpg"), [(1, 2, 3, 4)], [0.9], 1),
            DetectionResult(Path("b.jpg"), [(1, 2, 3, 4), (5, 6, 7, 8)], [0.8, 0.7], 2),
        ]

        stats = pipeline.get_statistics(results)
        assert stats["total_images"] == 2
        assert stats["total_swellings"] == 3
