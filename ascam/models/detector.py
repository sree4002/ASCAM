"""Object detection model for localizing axonal swellings."""

import logging
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from ultralytics import YOLO

from ascam.utils.image_utils import load_image, save_image
from ascam.utils.visualization import draw_boxes, add_count_label

logger = logging.getLogger(__name__)


class DetectionResult:
    """Container for detection results."""

    def __init__(
        self,
        image_path: Path,
        boxes: List[Tuple[int, int, int, int]],
        confidences: List[float],
        count: int,
    ):
        """
        Initialize detection result.

        Args:
            image_path: Path to the original image
            boxes: List of bounding boxes as (x1, y1, x2, y2)
            confidences: Confidence scores for each box
            count: Total number of detected swellings
        """
        self.image_path = image_path
        self.boxes = boxes
        self.confidences = confidences
        self.count = count

    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "image_name": self.image_path.name,
            "image_path": str(self.image_path),
            "count": self.count,
            "boxes": [
                {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[2]),
                    "y2": int(box[3]),
                    "confidence": float(self.confidences[i]),
                }
                for i, box in enumerate(self.boxes)
            ],
        }


class SwellingDetector:
    """
    YOLOv8s-based object detector for localizing axonal swellings.

    This model performs the second stage of the ASCAM pipeline by detecting
    and counting individual swellings in images. Uses imgsz=1280 for
    high-resolution inference and test-time augmentation (TTA) by default.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.50,
        max_detections: int = 1000,
        imgsz: int = 1280,
        augment: bool = True,
    ):
        """
        Initialize the detector.

        Args:
            model_path: Path to the trained YOLO model (.pt file)
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
            max_detections: Maximum number of detections per image
            imgsz: Inference image size (default: 1280)
            augment: Enable test-time augmentation (default: True)
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.imgsz = imgsz
        self.augment = augment
        self.model = None

        self._load_model()

    def _load_model(self):
        """Load the YOLO model from file."""
        try:
            self.model = YOLO(str(self.model_path))
            logger.info(f"Detector model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load detector model: {e}")
            raise

    def detect_single(
        self, image_path: Union[str, Path], verbose: bool = False
    ) -> DetectionResult:
        """
        Detect swellings in a single image.

        Args:
            image_path: Path to the image file
            verbose: Whether to print verbose output

        Returns:
            DetectionResult object containing boxes and counts
        """
        image_path = Path(image_path)

        try:
            # Run prediction
            results = self.model.predict(
                source=str(image_path),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections,
                imgsz=self.imgsz,
                augment=self.augment,
                verbose=verbose,
            )[0]

            # Extract boxes and confidences
            boxes = []
            confidences = []
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))
                confidences.append(float(box.conf[0]))

            count = len(boxes)

            logger.info(
                f"Image: {image_path.name} | "
                f"Detected {count} swellings "
                f"(conf≥{self.conf_threshold*100:.0f}%)"
            )

            return DetectionResult(
                image_path=image_path, boxes=boxes, confidences=confidences, count=count
            )

        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return DetectionResult(
                image_path=image_path, boxes=[], confidences=[], count=0
            )

    def detect_batch(
        self, image_paths: List[Union[str, Path]], verbose: bool = False
    ) -> List[DetectionResult]:
        """
        Detect swellings in multiple images.

        Args:
            image_paths: List of image file paths
            verbose: Whether to print verbose output

        Returns:
            List of DetectionResult objects
        """
        results = []
        for image_path in image_paths:
            result = self.detect_single(image_path, verbose=verbose)
            results.append(result)
        return results

    def detect_and_visualize(
        self,
        image_path: Union[str, Path],
        output_path: Union[str, Path],
        box_color: Tuple[int, int, int] = (0, 0, 255),
        box_thickness: int = 10,
        show_count: bool = True,
        show_confidence: bool = False,
    ) -> DetectionResult:
        """
        Detect swellings and save annotated image.

        Args:
            image_path: Path to input image
            output_path: Path to save annotated image
            box_color: Color for bounding boxes (BGR)
            box_thickness: Thickness of box lines
            show_count: Whether to show total count on image
            show_confidence: Whether to show confidence scores

        Returns:
            DetectionResult object
        """
        # Perform detection
        result = self.detect_single(image_path)

        if result.count == 0:
            logger.info(f"No swellings detected in {image_path}")
            return result

        # Load original image
        image = load_image(image_path)
        if image is None:
            logger.error(f"Failed to load image for visualization: {image_path}")
            return result

        # Draw boxes
        annotated = draw_boxes(
            image,
            result.boxes,
            color=box_color,
            thickness=box_thickness,
            confidences=result.confidences if show_confidence else None,
            show_conf=show_confidence,
        )

        # Add count label
        if show_count:
            annotated = add_count_label(
                annotated, result.count, color=box_color, thickness=box_thickness
            )

        # Save annotated image
        save_image(annotated, output_path)

        return result

    def validate(
        self,
        data_yaml: Optional[Union[str, Path]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> Dict:
        """
        Run validation on test dataset.

        Args:
            data_yaml: Path to data.yaml configuration file
            conf: Confidence threshold (uses model default if None)
            iou: IOU threshold (uses model default if None)

        Returns:
            Dictionary with validation metrics
        """
        conf = conf or self.conf_threshold
        iou = iou or self.iou_threshold

        try:
            metrics = self.model.val(
                data=str(data_yaml) if data_yaml else None,
                conf=conf,
                iou=iou,
                max_det=self.max_detections,
                imgsz=self.imgsz,
                plots=True,
                verbose=True,
            )

            results = {
                "precision": float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0.0,
                "recall": float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0.0,
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }

            logger.info(f"Validation results: {results}")
            return results

        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}
