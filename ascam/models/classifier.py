"""Classification model for swelling detection."""

import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Optional
import tensorflow as tf

from ascam.utils.image_utils import load_image, preprocess_image

logger = logging.getLogger(__name__)


class SwellingClassifier:
    """
    Binary classifier to determine if an image contains axonal swellings.

    This model performs the first stage of the ASCAM pipeline by filtering
    images as "swelling-positive" or "swelling-negative" to reduce computational
    load for the detection stage.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        image_size: Tuple[int, int] = (200, 200),
        threshold: float = 0.5
    ):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the trained Keras model (.keras file)
            image_size: Input image size as (height, width)
            threshold: Classification threshold (default 0.5)
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.threshold = threshold
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the Keras model from file."""
        try:
            self.model = tf.keras.models.load_model(str(self.model_path))
            logger.info(f"Classifier model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load classifier model: {e}")
            raise

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
        # Load and preprocess image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return (False, 0.0) if return_prob else False

        # Preprocess
        processed = preprocess_image(
            image,
            target_size=self.image_size,
            normalize=True,
            to_rgb=True
        )

        # Add batch dimension
        batch = np.expand_dims(processed, axis=0)

        # Predict
        try:
            prediction = self.model.predict(batch, verbose=0)
            prob = float(prediction[0][0])
            has_swelling = prob > self.threshold

            logger.info(
                f"Image: {Path(image_path).name} | "
                f"Probability: {prob:.4f} | "
                f"Prediction: {'Swelling' if has_swelling else 'No Swelling'}"
            )

            if return_prob:
                return has_swelling, prob
            return has_swelling

        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {e}")
            return (False, 0.0) if return_prob else False

    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        return_probs: bool = False
    ) -> Union[List[bool], List[Tuple[bool, float]]]:
        """
        Predict for multiple images.

        Args:
            image_paths: List of image file paths
            return_probs: If True, return probabilities with predictions

        Returns:
            List of predictions, or list of (prediction, probability) tuples
        """
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path, return_prob=return_probs)
            results.append(result)
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
        positive_images = []
        for image_path in image_paths:
            if self.predict_single(image_path):
                positive_images.append(Path(image_path))

        logger.info(
            f"Filtered {len(positive_images)} positive images "
            f"from {len(image_paths)} total images"
        )
        return positive_images


def build_classifier_model(
    image_height: int = 200,
    image_width: int = 200,
    channels: int = 3
) -> tf.keras.Model:
    """
    Build the CNN architecture for binary classification.

    Args:
        image_height: Input image height
        image_width: Input image width
        channels: Number of color channels

    Returns:
        Compiled Keras model
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu',
               input_shape=(image_height, image_width, channels)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model
