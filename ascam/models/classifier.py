"""Classification model for swelling detection using EfficientNet-B0."""

import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional

import torch
import torch.nn as nn
import timm
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    from torchvision import transforms
    ALBUMENTATIONS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class SwellingClassifier:
    """
    Binary classifier to determine if an image contains axonal swellings.

    This model performs the first stage of the ASCAM pipeline by filtering
    images as "swelling-positive" or "swelling-negative" to reduce computational
    load for the detection stage.

    Uses EfficientNet-B0 with transfer learning for robust classification.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: Tuple[int, int] = (224, 224),
        threshold: float = 0.5,
        model_name: str = "efficientnet_b0",
        num_classes: int = 2,
        device: Optional[str] = None
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the trained PyTorch model (.pt file)
            image_size: Input image size as (height, width)
            threshold: Classification threshold (default 0.5)
            model_name: timm model name (default: efficientnet_b0)
            num_classes: Number of output classes (default: 2)
            device: Device to use (auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.threshold = threshold
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device(
            device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.model = None
        self.transform = None

        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        """Load the PyTorch model from file."""
        try:
            self.model = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=self.num_classes
            )

            state_dict = torch.load(str(self.model_path), map_location=self.device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            self.model.load_state_dict(state_dict)

            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Classifier model loaded from {self.model_path} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise

    def _setup_transforms(self):
        """Setup inference transforms."""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if ALBUMENTATIONS_AVAILABLE:
            self.transform = A.Compose([
                A.Resize(self.image_size[0], self.image_size[1]),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def _preprocess_image(self, image_path: Union[str, Path]) -> Optional[torch.Tensor]:
        """Load and preprocess a single image."""
        try:
            image = Image.open(str(image_path)).convert('RGB')
            image_np = np.array(image)

            if ALBUMENTATIONS_AVAILABLE:
                transformed = self.transform(image=image_np)
                tensor = transformed['image']
            else:
                tensor = self.transform(image)

            return tensor
        except Exception as e:
            logger.error(f"Failed to preprocess image {image_path}: {e}")
            return None

    @torch.no_grad()
    def predict_single(
        self,
        image_path: Union[str, Path],
        return_prob: bool = False
    ) -> Union[bool, Tuple[bool, float]]:
        """
        Predict if a single image contains swellings.

        Args:
            image_path: Path to the image file
            return_prob: If True, return (prediction, probability)

        Returns:
            Boolean prediction, or (prediction, probability) if return_prob=True
        """
        tensor = self._preprocess_image(image_path)
        if tensor is None:
            return (False, 0.0) if return_prob else False

        batch = tensor.unsqueeze(0).to(self.device)

        output = self.model(batch)
        probs = torch.softmax(output, dim=1)

        # class 1 = swelling, class 0 = no_swelling
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
        has_swelling = (pred_class == 1)

        logger.info(
            f"Image: {Path(image_path).name} | "
            f"Confidence: {confidence:.4f} | "
            f"Prediction: {'Swelling' if has_swelling else 'No Swelling'}"
        )

        if return_prob:
            return has_swelling, confidence
        return has_swelling

    @torch.no_grad()
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        return_probs: bool = False,
        batch_size: int = 32
    ) -> Union[List[bool], List[Tuple[bool, float]]]:
        """
        Predict for multiple images using true batched inference.

        Args:
            image_paths: List of image file paths
            return_probs: If True, return probabilities with predictions
            batch_size: Number of images to process at once

        Returns:
            List of predictions, or list of (prediction, probability) tuples
        """
        results = []

        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i:i + batch_size]
            tensors = []
            valid_indices = []

            for j, path in enumerate(chunk_paths):
                tensor = self._preprocess_image(path)
                if tensor is not None:
                    tensors.append(tensor)
                    valid_indices.append(j)
                else:
                    # Placeholder for failed images
                    pass

            if tensors:
                batch = torch.stack(tensors).to(self.device)
                outputs = self.model(batch)
                probs = torch.softmax(outputs, dim=1)
                pred_classes = probs.argmax(dim=1)

                tensor_idx = 0
                for j in range(len(chunk_paths)):
                    if j in valid_indices:
                        pred_class = pred_classes[tensor_idx].item()
                        confidence = probs[tensor_idx, pred_class].item()
                        has_swelling = (pred_class == 1)
                        tensor_idx += 1

                        if return_probs:
                            results.append((has_swelling, confidence))
                        else:
                            results.append(has_swelling)
                    else:
                        if return_probs:
                            results.append((False, 0.0))
                        else:
                            results.append(False)
            else:
                for _ in chunk_paths:
                    if return_probs:
                        results.append((False, 0.0))
                    else:
                        results.append(False)

        return results

    def filter_positive_images(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[Path]:
        """
        Filter images that are predicted to contain swellings.

        Args:
            image_paths: List of image paths to filter

        Returns:
            List of paths for images predicted to contain swellings
        """
        predictions = self.predict_batch(image_paths)
        positive_images = [
            Path(path) for path, pred in zip(image_paths, predictions) if pred
        ]

        logger.info(
            f"Filtered {len(positive_images)} positive images "
            f"from {len(image_paths)} total images"
        )
        return positive_images


def build_classifier_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3
) -> nn.Module:
    """
    Build the EfficientNet-B0 model for binary classification.

    Args:
        model_name: timm model name
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate

    Returns:
        PyTorch model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout
    )
    return model
