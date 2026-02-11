"""Tests for the SwellingClassifier."""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from ascam.models.classifier import (
    SwellingClassifier,
    FocalLoss,
    build_classifier_model,
)


class TestFocalLoss:
    """Tests for FocalLoss."""

    def test_focal_loss_output_shape(self):
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        inputs = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        loss = loss_fn(inputs, targets)
        assert loss.ndim == 0  # scalar

    def test_focal_loss_reductions(self):
        inputs = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss_mean = FocalLoss(reduction='mean')(inputs, targets)
        loss_sum = FocalLoss(reduction='sum')(inputs, targets)
        loss_none = FocalLoss(reduction='none')(inputs, targets)

        assert loss_mean.ndim == 0
        assert loss_sum.ndim == 0
        assert loss_none.shape == (4,)

    def test_focal_loss_non_negative(self):
        loss_fn = FocalLoss()
        inputs = torch.randn(10, 2)
        targets = torch.randint(0, 2, (10,))
        loss = loss_fn(inputs, targets)
        assert loss.item() >= 0


class TestBuildClassifierModel:
    """Tests for build_classifier_model."""

    def test_model_creation(self):
        model = build_classifier_model(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False,
            dropout=0.3
        )
        assert isinstance(model, torch.nn.Module)

    def test_model_output_shape(self):
        model = build_classifier_model(
            model_name="efficientnet_b0",
            num_classes=2,
            pretrained=False
        )
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        assert output.shape == (1, 2)

    def test_model_num_classes(self):
        for n_classes in [2, 5, 10]:
            model = build_classifier_model(
                model_name="efficientnet_b0",
                num_classes=n_classes,
                pretrained=False
            )
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, n_classes)


class TestSwellingClassifier:
    """Tests for SwellingClassifier with mocked model loading."""

    @patch('ascam.models.classifier.timm.create_model')
    @patch('ascam.models.classifier.torch.load')
    def test_predict_single(self, mock_load, mock_create, sample_image):
        # Setup mock model
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        output = torch.tensor([[0.8, 0.2]])
        mock_model.__call__ = MagicMock(return_value=output)
        mock_model.return_value = output
        mock_create.return_value = mock_model
        mock_load.return_value = {}

        classifier = SwellingClassifier(
            model_path=sample_image,  # dummy path
            device='cpu'
        )

        # Mock the model call
        classifier.model = mock_model

        result = classifier.predict_single(sample_image)
        assert isinstance(result, bool)

    @patch('ascam.models.classifier.timm.create_model')
    @patch('ascam.models.classifier.torch.load')
    def test_predict_single_with_prob(self, mock_load, mock_create, sample_image):
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        output = torch.tensor([[0.3, 0.7]])
        mock_model.return_value = output
        mock_create.return_value = mock_model
        mock_load.return_value = {}

        classifier = SwellingClassifier(
            model_path=sample_image,
            device='cpu'
        )
        classifier.model = mock_model

        result = classifier.predict_single(sample_image, return_prob=True)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    @patch('ascam.models.classifier.timm.create_model')
    @patch('ascam.models.classifier.torch.load')
    def test_predict_batch(self, mock_load, mock_create, sample_image_dir):
        mock_model = MagicMock()
        mock_model.eval = MagicMock(return_value=mock_model)
        mock_model.to = MagicMock(return_value=mock_model)
        # Return batch output
        output = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9], [0.5, 0.5]])
        mock_model.return_value = output
        mock_create.return_value = mock_model
        mock_load.return_value = {}

        classifier = SwellingClassifier(
            model_path=list(sample_image_dir.iterdir())[0],
            device='cpu'
        )
        classifier.model = mock_model

        image_files = list(sample_image_dir.iterdir())
        results = classifier.predict_batch(image_files[:5])
        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)
