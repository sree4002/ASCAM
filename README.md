# ASCAM

**Axonal Swelling Classification and Analysis Model**
Automated detection and quantification of axonal swellings in histological brain sections using deep learning.

[![CI](https://github.com/sree4002/ASCAM/actions/workflows/ci.yml/badge.svg)](https://github.com/sree4002/ASCAM/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Language: Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![Model Type](https://img.shields.io/badge/model-YOLOv8s%20%2B%20EfficientNet--B0-orange)

---

## Overview

ASCAM is a two-stage deep learning pipeline designed for **automated detection and quantification of axonal swellings**, a pathological marker in neurodegenerative diseases like Alzheimer's disease. The system analyzes DAB-stained microscopy images from 5xFAD mouse brain sections.

### Two-Stage Pipeline

1. **Classification Model (Stage 1)** -- EfficientNet-B0 (transfer learning) classifier that filters images as "swelling-positive" or "swelling-negative" to reduce computational load. Input: 224x224 RGB.
2. **Detection Model (Stage 2)** -- YOLOv8s object detector that localizes and counts individual axonal swellings with bounding boxes. Uses imgsz=1280, conf=0.25, iou=0.50 with test-time augmentation (TTA).

ASCAM dramatically reduces analysis time and variability compared to manual quantification, providing a scalable solution for histological analysis in neuroscience research.

---

## Repository Structure

```
ASCAM/
├── ascam/                      # Python package
│   ├── models/                 # Model classes
│   │   ├── classifier.py       # EfficientNet-B0 classification model
│   │   └── detector.py         # YOLOv8s detection model
│   ├── training/               # Training module
│   │   ├── train_cnn.py        # CNN training with Focal Loss, OneCycleLR
│   │   ├── train_yolo.py       # YOLO training (paper-compliant)
│   │   ├── augment.py          # Data augmentation
│   │   └── evaluate.py         # Threshold calibration & metrics
│   ├── utils/                  # Utility functions
│   ├── pipeline.py             # Complete pipeline
│   ├── config.py               # Configuration management
│   └── cli.py                  # Command-line interface
├── models/                     # Trained model weights
│   ├── classifier.pt           # EfficientNet-B0 classifier (16 MB)
│   └── yolov8s_best.pt         # YOLOv8s detection model (23 MB)
├── configs/                    # Configuration files
│   └── default_config.yaml     # Default settings
├── tests/                      # Test suite
├── scripts/                    # Reproducibility scripts
│   └── reproduce.sh            # One-command end-to-end pipeline
├── test_images/                # Sample test images
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── pyproject.toml              # Modern package metadata
```

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Quick Install

```bash
# Clone the repository
git clone https://github.com/sree4002/ASCAM.git
cd ASCAM

# Install the package
pip install -e .

# Verify installation
ascam --version
```

### Install with Optional Dependencies

```bash
# For development (testing, linting)
pip install -e ".[dev]"

# For training models
pip install -e ".[training]"

# For advanced augmentation
pip install -e ".[augmentation]"
```

---

## Usage

### Command-Line Interface (CLI)

ASCAM provides a comprehensive CLI for all operations:

#### 1. Full Pipeline (Recommended)

Process images through both classification and detection:

```bash
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --classifier models/classifier.pt \
  --detector models/yolov8s_best.pt
```

#### 2. Classification Only

Filter images for presence of swellings:

```bash
ascam classify \
  --input test_images/ \
  --model models/classifier.pt \
  --threshold 0.5
```

#### 3. Detection Only

Detect and count swellings (skip classification):

```bash
ascam detect \
  --input test_images/ \
  --output results/ \
  --model models/yolov8s_best.pt \
  --conf 0.25
```

#### 4. Train Models

```bash
# Train CNN classifier (EfficientNet-B0)
ascam train-cnn --data data/classifier/ --epochs 50 --output models/

# Train YOLO detector (paper-compliant parameters)
ascam train-yolo --data data/detection/data.yaml --epochs 500 --output models/
```

#### 5. Using Configuration Files

```bash
# Generate default config
ascam config --output my_config.yaml

# Run with custom config
ascam pipeline --input test_images/ --output results/ --config my_config.yaml
```

### Advanced Options

```bash
# Adjust detection confidence threshold
ascam pipeline --input images/ --output results/ --conf 0.25

# Skip classification stage (faster)
ascam pipeline --input images/ --output results/ --skip-classify

# Save results as CSV instead of JSON
ascam pipeline --input images/ --output results/ --format csv

# Show confidence scores on boxes
ascam detect --input images/ --output results/ --model models/yolov8s_best.pt --show-conf

# Verbose output for debugging
ascam pipeline --input images/ --output results/ --verbose
```

---

## Python API

Use ASCAM directly in your Python scripts:

```python
from ascam import ASCAMPipeline, SwellingClassifier, SwellingDetector

# Initialize the full pipeline
pipeline = ASCAMPipeline(
    classifier_path="models/classifier.pt",
    detector_path="models/yolov8s_best.pt",
    classifier_threshold=0.5,
    detector_conf=0.25
)

# Process a directory
results = pipeline.process_directory(
    input_dir="test_images/",
    output_dir="results/",
    visualize=True,
    save_results=True
)

# Get statistics
stats = pipeline.get_statistics(results)
print(f"Total swellings detected: {stats['total_swellings']}")
print(f"Mean per image: {stats['mean_swellings_per_image']:.2f}")

# Use models independently
classifier = SwellingClassifier("models/classifier.pt")
has_swelling = classifier.predict_single("image.jpg")

detector = SwellingDetector("models/yolov8s_best.pt")
result = detector.detect_and_visualize("image.jpg", "output.jpg")
print(f"Detected {result.count} swellings")
```

---

## Output Formats

### JSON Results (`results.json`)

```json
{
  "total_images": 12,
  "total_swellings": 173,
  "results": [
    {
      "image_name": "IMG_0022.JPG",
      "image_path": "test_images/IMG_0022.JPG",
      "count": 13,
      "boxes": [
        {"x1": 1234, "y1": 567, "x2": 1345, "y2": 678, "confidence": 0.85}
      ]
    }
  ]
}
```

### CSV Results (`results.csv`)

```
Image Name,Image Path,Swelling Count,Average Confidence
IMG_0022.JPG,test_images/IMG_0022.JPG,13,0.7234
```

---

## Configuration

Default configuration (`configs/default_config.yaml`):

```yaml
classifier:
  model_path: models/classifier.pt
  model_name: efficientnet_b0
  image_size: [224, 224]
  threshold: 0.5

detector:
  model_path: models/yolov8s_best.pt
  model_name: yolov8s
  imgsz: 1280
  conf_threshold: 0.25
  iou_threshold: 0.5
  augment: true

pipeline:
  skip_classification: false
  save_results: true
  results_format: json
```

---

## 🔬 Model Information

### Classification Model (Stage 1)
- **Architecture:** EfficientNet-B0 (transfer learning from ImageNet)
- **Input:** 224×224 RGB images
- **Output:** Binary (swelling/no swelling)
- **Loss Function:** Focal Loss (handles class imbalance)
- **Optimizer:** AdamW with OneCycleLR scheduler
- **Best Validation Accuracy:** 87.9%
- **Test Accuracy:** 71.7%
- **AUC-ROC:** 0.857
- **Swelling Recall:** 83.3% (prioritized for filtering)
- **File:** `models/classifier.pt` (16 MB)

### Detection Model (Stage 2)
- **Architecture:** YOLOv8s
- **Input:** 1280×1280 pixels
- **Inference:** Test-time augmentation (TTA) enabled
- **Confidence Threshold:** 0.25
- **IoU Threshold:** 0.5
- **Precision:** 71.0%
- **Recall:** 64.9%
- **mAP@0.5:** 71.0%
- **F1 Score:** 67.8%
- **File:** `models/yolov8s_best.pt` (23 MB)

---

## Reproducibility

For full end-to-end reproducibility (requires training data):

```bash
bash scripts/reproduce.sh
```

This script trains both models from scratch, evaluates thresholds, and runs the full pipeline.

### Expected Data Structure

**CNN classifier data:**
```
data/classifier/
├── swelling/          # positive samples
└── no_swelling/       # negative samples
```

**YOLO detector data:**
```
data/detection/
├── train/images/
├── train/labels/
├── val/images/
├── val/labels/
├── test/images/
├── test/labels/
└── data.yaml
```

---

## Testing

```bash
# Run test suite
pytest tests/ -v

# Run with sample images
ascam pipeline \
  --input test_images/ \
  --output test_results/ \
  --classifier models/classifier.pt \
  --detector models/yolov8s_best.pt
```

---

## Citation

If you use ASCAM in your research, please cite:

```bibtex
@software{ascam,
  title = {ASCAM: Axonal Swelling Classification and Analysis Model},
  author = {ASCAM Team},
  year = {2024},
  url = {https://github.com/sree4002/ASCAM}
}
```

See [CITATION.cff](CITATION.cff) for the full citation metadata.

---

## Contributing

Thank you for wanting to improve ASCAM! If you encounter any bugs, have questions, or would like to contribute new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
