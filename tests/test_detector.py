"""Tests for the SwellingDetector."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from ascam.models.detector import SwellingDetector, DetectionResult


class TestDetectionResult:
    """Tests for DetectionResult dataclass."""

    def test_creation(self):
        result = DetectionResult(
            image_path=Path("test.jpg"),
            boxes=[(10, 20, 30, 40)],
            confidences=[0.95],
            count=1
        )
        assert result.count == 1
        assert len(result.boxes) == 1
        assert len(result.confidences) == 1

    def test_to_dict(self):
        result = DetectionResult(
            image_path=Path("test.jpg"),
            boxes=[(10, 20, 30, 40), (50, 60, 70, 80)],
            confidences=[0.95, 0.85],
            count=2
        )
        d = result.to_dict()
        assert d["image_name"] == "test.jpg"
        assert d["count"] == 2
        assert len(d["boxes"]) == 2
        assert d["boxes"][0]["x1"] == 10
        assert d["boxes"][0]["confidence"] == 0.95

    def test_empty_result(self):
        result = DetectionResult(
            image_path=Path("test.jpg"),
            boxes=[],
            confidences=[],
            count=0
        )
        assert result.count == 0
        d = result.to_dict()
        assert d["count"] == 0
        assert len(d["boxes"]) == 0


class TestSwellingDetector:
    """Tests for SwellingDetector with mocked YOLO model."""

    @patch('ascam.models.detector.YOLO')
    def test_initialization(self, mock_yolo):
        mock_yolo.return_value = MagicMock()
        detector = SwellingDetector(
            model_path="yolov8s_best.pt",
            conf_threshold=0.25,
            iou_threshold=0.50
        )
        assert detector.conf_threshold == 0.25
        assert detector.iou_threshold == 0.50

    @patch('ascam.models.detector.YOLO')
    def test_detect_single(self, mock_yolo, sample_image):
        # Mock YOLO prediction results
        mock_box = MagicMock()
        mock_box.xyxy = [[10, 20, 30, 40]]
        mock_box.conf = [0.95]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]

        mock_model = MagicMock()
        mock_model.predict.return_value = [mock_result]
        mock_yolo.return_value = mock_model

        detector = SwellingDetector(model_path="yolov8s_best.pt")
        result = detector.detect_single(sample_image)

        assert isinstance(result, DetectionResult)
        assert result.count >= 0
