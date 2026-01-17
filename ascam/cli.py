"""Command-line interface for ASCAM."""

import argparse
import logging
import sys
from pathlib import Path

from ascam import __version__
from ascam.models.classifier import SwellingClassifier
from ascam.models.detector import SwellingDetector
from ascam.pipeline import ASCAMPipeline
from ascam.config import Config, create_default_config
from ascam.utils.image_utils import get_image_files


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def cmd_classify(args):
    """Run classification on images."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Running classification...")

    # Initialize classifier
    classifier = SwellingClassifier(
        model_path=args.model,
        threshold=args.threshold
    )

    # Get input images
    if Path(args.input).is_dir():
        image_files = get_image_files(args.input)
    else:
        image_files = [Path(args.input)]

    if not image_files:
        logger.error(f"No images found in {args.input}")
        sys.exit(1)

    # Process images
    results = classifier.predict_batch(image_files, return_probs=True)

    # Print results
    logger.info("\nClassification Results:")
    logger.info("-" * 60)
    positive_count = 0
    for img, (has_swelling, prob) in zip(image_files, results):
        label = "Swelling" if has_swelling else "No Swelling"
        logger.info(f"{img.name:30} | {label:12} | {prob:.4f}")
        if has_swelling:
            positive_count += 1

    logger.info("-" * 60)
    logger.info(f"Total: {len(image_files)} | Positive: {positive_count} | "
                f"Negative: {len(image_files) - positive_count}")


def cmd_detect(args):
    """Run detection on images."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Running detection...")

    # Initialize detector
    detector = SwellingDetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        max_detections=args.max_det
    )

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get input images
    if Path(args.input).is_dir():
        image_files = get_image_files(args.input)
    else:
        image_files = [Path(args.input)]

    if not image_files:
        logger.error(f"No images found in {args.input}")
        sys.exit(1)

    # Process images
    logger.info(f"Processing {len(image_files)} images...")
    total_swellings = 0

    for img_path in image_files:
        result = detector.detect_and_visualize(
            image_path=img_path,
            output_path=output_dir / img_path.name,
            show_count=not args.no_count,
            show_confidence=args.show_conf
        )
        total_swellings += result.count

    logger.info(f"\nDetection complete!")
    logger.info(f"Total swellings detected: {total_swellings}")
    logger.info(f"Results saved to: {output_dir}")


def cmd_pipeline(args):
    """Run full pipeline on images."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Running ASCAM pipeline...")

    # Load configuration if provided
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()

    # Override config with command-line arguments
    if args.classifier:
        config.set('classifier.model_path', args.classifier)
    if args.detector:
        config.set('detector.model_path', args.detector)
    if args.skip_classify:
        config.set('pipeline.skip_classification', True)

    # Initialize pipeline
    pipeline = ASCAMPipeline(
        classifier_path=config.get('classifier.model_path'),
        detector_path=config.get('detector.model_path'),
        classifier_threshold=config.get('classifier.threshold', 0.5),
        detector_conf=args.conf if args.conf else config.get('detector.conf_threshold', 0.02),
        detector_iou=args.iou if args.iou else config.get('detector.iou_threshold', 0.30),
        skip_classification=config.get('pipeline.skip_classification', False)
    )

    # Process directory
    results = pipeline.process_directory(
        input_dir=args.input,
        output_dir=args.output,
        visualize=not args.no_visualize,
        save_results=True,
        results_format=args.format
    )

    # Print statistics
    if results:
        stats = pipeline.get_statistics(results)
        logger.info("\nStatistics:")
        logger.info("-" * 60)
        for key, value in stats.items():
            logger.info(f"{key:30}: {value:.2f}" if isinstance(value, float) else f"{key:30}: {value}")


def cmd_config(args):
    """Generate default configuration file."""
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    output_path = Path(args.output)
    create_default_config(output_path)
    logger.info(f"Default configuration created at: {output_path}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="ASCAM - Axonal Swelling Classification and Analysis Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  ascam pipeline --input images/ --output results/ --classifier models/best_model.keras --detector models/weights.pt

  # Run classification only
  ascam classify --input images/ --model models/best_model.keras

  # Run detection only
  ascam detect --input images/ --output results/ --model models/weights.pt

  # Generate default config
  ascam config --output config.yaml
        """
    )

    parser.add_argument('--version', action='version', version=f'ASCAM {__version__}')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Classify command
    classify_parser = subparsers.add_parser(
        'classify',
        help='Run binary classification on images'
    )
    classify_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input image or directory'
    )
    classify_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to classification model (.keras)'
    )
    classify_parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )
    classify_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    classify_parser.set_defaults(func=cmd_classify)

    # Detect command
    detect_parser = subparsers.add_parser(
        'detect',
        help='Run object detection on images'
    )
    detect_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input image or directory'
    )
    detect_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    detect_parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to detection model (.pt)'
    )
    detect_parser.add_argument(
        '--conf',
        type=float,
        default=0.02,
        help='Confidence threshold (default: 0.02)'
    )
    detect_parser.add_argument(
        '--iou',
        type=float,
        default=0.30,
        help='IOU threshold for NMS (default: 0.30)'
    )
    detect_parser.add_argument(
        '--max-det',
        type=int,
        default=1000,
        help='Maximum detections per image (default: 1000)'
    )
    detect_parser.add_argument(
        '--no-count',
        action='store_true',
        help='Do not show count on images'
    )
    detect_parser.add_argument(
        '--show-conf',
        action='store_true',
        help='Show confidence scores on boxes'
    )
    detect_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    detect_parser.set_defaults(func=cmd_detect)

    # Pipeline command
    pipeline_parser = subparsers.add_parser(
        'pipeline',
        help='Run complete classification + detection pipeline'
    )
    pipeline_parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input directory containing images'
    )
    pipeline_parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for results'
    )
    pipeline_parser.add_argument(
        '--config', '-c',
        help='Path to configuration YAML file'
    )
    pipeline_parser.add_argument(
        '--classifier',
        help='Path to classification model (overrides config)'
    )
    pipeline_parser.add_argument(
        '--detector',
        help='Path to detection model (overrides config)'
    )
    pipeline_parser.add_argument(
        '--skip-classify',
        action='store_true',
        help='Skip classification stage'
    )
    pipeline_parser.add_argument(
        '--conf',
        type=float,
        help='Detection confidence threshold (overrides config)'
    )
    pipeline_parser.add_argument(
        '--iou',
        type=float,
        help='Detection IOU threshold (overrides config)'
    )
    pipeline_parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Do not save annotated images'
    )
    pipeline_parser.add_argument(
        '--format', '-f',
        choices=['json', 'csv'],
        default='json',
        help='Results file format (default: json)'
    )
    pipeline_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    pipeline_parser.set_defaults(func=cmd_pipeline)

    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Generate default configuration file'
    )
    config_parser.add_argument(
        '--output', '-o',
        default='config.yaml',
        help='Output path for config file (default: config.yaml)'
    )
    config_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    config_parser.set_defaults(func=cmd_config)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    args.func(args)


if __name__ == '__main__':
    main()
