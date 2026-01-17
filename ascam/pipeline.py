"""Complete ASCAM pipeline combining classification and detection."""

import logging
import json
import csv
from pathlib import Path
from typing import Union, List, Dict, Optional
from tqdm import tqdm

from ascam.models.classifier import SwellingClassifier
from ascam.models.detector import SwellingDetector, DetectionResult
from ascam.utils.image_utils import get_image_files

logger = logging.getLogger(__name__)


class ASCAMPipeline:
    """
    Complete two-stage pipeline for axonal swelling analysis.

    Stage 1: Binary classification to filter images containing swellings
    Stage 2: Object detection to localize and count individual swellings
    """

    def __init__(
        self,
        classifier_path: Union[str, Path],
        detector_path: Union[str, Path],
        classifier_threshold: float = 0.5,
        detector_conf: float = 0.02,
        detector_iou: float = 0.30,
        skip_classification: bool = False
    ):
        """
        Initialize the ASCAM pipeline.

        Args:
            classifier_path: Path to classification model (.keras)
            detector_path: Path to detection model (.pt)
            classifier_threshold: Classification threshold
            detector_conf: Detection confidence threshold
            detector_iou: Detection IOU threshold
            skip_classification: If True, skip classification stage
        """
        self.skip_classification = skip_classification

        # Initialize classifier
        if not skip_classification:
            self.classifier = SwellingClassifier(
                model_path=classifier_path,
                threshold=classifier_threshold
            )
        else:
            self.classifier = None
            logger.info("Classification stage disabled")

        # Initialize detector
        self.detector = SwellingDetector(
            model_path=detector_path,
            conf_threshold=detector_conf,
            iou_threshold=detector_iou
        )

    def process_single(
        self,
        image_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        visualize: bool = True
    ) -> Optional[DetectionResult]:
        """
        Process a single image through the pipeline.

        Args:
            image_path: Path to input image
            output_dir: Directory to save results (if visualize=True)
            visualize: Whether to save annotated image

        Returns:
            DetectionResult if swellings detected, None otherwise
        """
        image_path = Path(image_path)

        # Stage 1: Classification
        if not self.skip_classification:
            has_swelling, prob = self.classifier.predict_single(
                image_path,
                return_prob=True
            )

            if not has_swelling:
                logger.info(
                    f"Image {image_path.name} classified as negative "
                    f"(prob={prob:.4f}), skipping detection"
                )
                return None

        # Stage 2: Detection
        if visualize and output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / image_path.name

            result = self.detector.detect_and_visualize(
                image_path=image_path,
                output_path=output_path
            )
        else:
            result = self.detector.detect_single(image_path)

        return result

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        visualize: bool = True,
        save_results: bool = True,
        results_format: str = "json"
    ) -> List[DetectionResult]:
        """
        Process all images in a directory.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save results
            visualize: Whether to save annotated images
            save_results: Whether to save results file
            results_format: Format for results file ('json' or 'csv')

        Returns:
            List of DetectionResult objects
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = get_image_files(input_dir)
        if not image_files:
            logger.warning(f"No images found in {input_dir}")
            return []

        logger.info(f"Processing {len(image_files)} images from {input_dir}")

        # Process each image
        all_results = []
        for image_path in tqdm(image_files, desc="Processing images"):
            result = self.process_single(
                image_path=image_path,
                output_dir=output_dir if visualize else None,
                visualize=visualize
            )
            if result:
                all_results.append(result)

        # Save results summary
        if save_results and all_results:
            if results_format.lower() == "json":
                self._save_results_json(all_results, output_dir / "results.json")
            elif results_format.lower() == "csv":
                self._save_results_csv(all_results, output_dir / "results.csv")

        # Print summary
        total_images = len(image_files)
        positive_images = len(all_results)
        total_swellings = sum(r.count for r in all_results)

        logger.info(
            f"\nPipeline Summary:\n"
            f"  Total images processed: {total_images}\n"
            f"  Images with swellings: {positive_images}\n"
            f"  Total swellings detected: {total_swellings}\n"
            f"  Average swellings per positive image: "
            f"{total_swellings/positive_images if positive_images > 0 else 0:.2f}"
        )

        return all_results

    def _save_results_json(
        self,
        results: List[DetectionResult],
        output_path: Path
    ):
        """Save results to JSON file."""
        data = {
            "total_images": len(results),
            "total_swellings": sum(r.count for r in results),
            "results": [r.to_dict() for r in results]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def _save_results_csv(
        self,
        results: List[DetectionResult],
        output_path: Path
    ):
        """Save results to CSV file."""
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Image Name",
                "Image Path",
                "Swelling Count",
                "Average Confidence"
            ])

            for result in results:
                avg_conf = (
                    sum(result.confidences) / len(result.confidences)
                    if result.confidences else 0.0
                )
                writer.writerow([
                    result.image_path.name,
                    str(result.image_path),
                    result.count,
                    f"{avg_conf:.4f}"
                ])

        logger.info(f"Results saved to {output_path}")

    def get_statistics(
        self,
        results: List[DetectionResult]
    ) -> Dict:
        """
        Calculate statistics from detection results.

        Args:
            results: List of DetectionResult objects

        Returns:
            Dictionary with statistics
        """
        if not results:
            return {}

        counts = [r.count for r in results]
        confidences = [c for r in results for c in r.confidences]

        import numpy as np

        stats = {
            "total_images": len(results),
            "total_swellings": sum(counts),
            "mean_swellings_per_image": np.mean(counts),
            "median_swellings_per_image": np.median(counts),
            "std_swellings_per_image": np.std(counts),
            "min_swellings": min(counts),
            "max_swellings": max(counts),
        }

        if confidences:
            stats.update({
                "mean_confidence": np.mean(confidences),
                "median_confidence": np.median(confidences),
                "min_confidence": min(confidences),
                "max_confidence": max(confidences),
            })

        return stats
