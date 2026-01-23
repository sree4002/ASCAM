# ASCAM

**Axonal Swelling Classification and Analysis Model**
Automated detection and quantification of axonal swellings in histological brain sections using deep learning.

[![Language: Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)
![Model Type](https://img.shields.io/badge/model-YOLOv8%20%2B%20CNN-orange)

---

## 🧠 Overview

ASCAM is a two-stage deep learning pipeline designed for **automated detection and quantification of axonal swellings**, a pathological marker in neurodegenerative diseases like Alzheimer's disease. The system analyzes DAB-stained microscopy images from 5xFAD mouse brain sections.

### Two-Stage Pipeline

1. **Classification Model (Stage 1)** – Binary CNN classifier that filters images as "swelling-positive" or "swelling-negative" to reduce computational load.
2. **Detection Model (Stage 2)** – YOLOv8-based object detector that localizes and counts individual axonal swellings with bounding boxes.

ASCAM dramatically reduces analysis time and variability compared to manual quantification, providing a scalable solution for histological analysis in neuroscience research.

---

## 📁 Repository Structure

```
ASCAM/
├── ascam/                      # Python package
│   ├── models/                 # Model classes
│   │   ├── classifier.py       # Classification model
│   │   └── detector.py         # Detection model
│   ├── utils/                  # Utility functions
│   ├── pipeline.py             # Complete pipeline
│   ├── config.py               # Configuration management
│   └── cli.py                  # Command-line interface
├── models/                     # Trained model weights
│   ├── best_model.keras        # Classification model (162 MB)
│   └── weights.pt              # YOLO detection model (22.5 MB)
├── configs/                    # Configuration files
│   └── default_config.yaml     # Default settings
├── Classification_Model/       # Training notebooks (legacy)
├── Object_Detection_Model/     # Training notebooks (legacy)
├── Running_Code/               # Inference notebooks (legacy)
├── test_images/                # Sample test images
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
└── pyproject.toml              # Modern package metadata
```

---

## 🚀 Installation

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

### Download Model Files (Git LFS)

The trained model weights are stored using [Git Large File Storage (LFS)](https://git-lfs.com/). After cloning the repository, you need to download the actual model files:

```bash
# Install Git LFS (if not already installed)
# macOS
brew install git-lfs

# Ubuntu/Debian
sudo apt-get install git-lfs

# Windows (using Chocolatey)
choco install git-lfs

# Initialize Git LFS
git lfs install

# Download the model files
cd ASCAM
git lfs pull
```

**Note:** Without Git LFS, the model files (`models/best_model.keras` and `models/weights.pt`) will be small pointer files instead of the actual models, causing errors when running the pipeline.

### Install from PyPI (when published)

```bash
pip install ascam
```

### Install with Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For training models
pip install -e ".[training]"
```

---

## 💻 Usage

### Command-Line Interface (CLI)

ASCAM provides a comprehensive CLI for all operations:

#### 1. Full Pipeline (Recommended)

Process images through both classification and detection:

```bash
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --classifier models/best_model.keras \
  --detector models/weights.pt
```

**Output:**
- Annotated images with bounding boxes in `results/`
- `results.json` with detection counts and statistics
- Summary statistics in terminal

#### 2. Classification Only

Filter images for presence of swellings:

```bash
ascam classify \
  --input test_images/ \
  --model models/best_model.keras \
  --threshold 0.5
```

#### 3. Detection Only

Detect and count swellings (skip classification):

```bash
ascam detect \
  --input test_images/ \
  --output results/ \
  --model models/weights.pt \
  --conf 0.02
```

#### 4. Using Configuration Files

Create and use a config file:

```bash
# Generate default config
ascam config --output my_config.yaml

# Edit my_config.yaml as needed, then run:
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --config my_config.yaml
```

### Advanced Options

```bash
# Skip classification stage (faster, less accurate filtering)
ascam pipeline --input images/ --output results/ --skip-classify

# Save results as CSV instead of JSON
ascam pipeline --input images/ --output results/ --format csv

# Adjust detection confidence threshold
ascam pipeline --input images/ --output results/ --conf 0.05

# Show confidence scores on boxes
ascam detect --input images/ --output results/ --model models/weights.pt --show-conf

# Verbose output for debugging
ascam pipeline --input images/ --output results/ --verbose
```

---

## 🐍 Python API

Use ASCAM directly in your Python scripts:

```python
from ascam import ASCAMPipeline, SwellingClassifier, SwellingDetector

# Initialize the full pipeline
pipeline = ASCAMPipeline(
    classifier_path="models/best_model.keras",
    detector_path="models/weights.pt",
    classifier_threshold=0.5,
    detector_conf=0.02
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
classifier = SwellingClassifier("models/best_model.keras")
has_swelling = classifier.predict_single("image.jpg")

detector = SwellingDetector("models/weights.pt")
result = detector.detect_and_visualize("image.jpg", "output.jpg")
print(f"Detected {result.count} swellings")
```

---

## 📊 Output Formats

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
        {"x1": 1234, "y1": 567, "x2": 1345, "y2": 678, "confidence": 0.85},
        ...
      ]
    },
    ...
  ]
}
```

### CSV Results (`results.csv`)

```
Image Name,Image Path,Swelling Count,Average Confidence
IMG_0022.JPG,test_images/IMG_0022.JPG,13,0.7234
IMG_0023.JPG,test_images/IMG_0023.JPG,16,0.6891
...
```

---

## ⚙️ Configuration

Default configuration file (`configs/default_config.yaml`):

```yaml
classifier:
  model_path: models/best_model.keras
  image_size: [200, 200]
  threshold: 0.5

detector:
  model_path: models/weights.pt
  conf_threshold: 0.02
  iou_threshold: 0.30
  max_detections: 1000

visualization:
  box_color: [0, 0, 255]  # BGR: Red
  box_thickness: 10
  show_count: true
  show_confidence: false

pipeline:
  skip_classification: false
  save_results: true
  results_format: json
```

---

## 🔬 Model Information

### Classification Model
- **Architecture:** CNN with batch normalization
- **Input:** 200×200 RGB images
- **Output:** Binary (swelling/no swelling)
- **Training:** 100 epochs with early stopping
- **File:** `best_model.keras` (162 MB)

### Detection Model
- **Architecture:** YOLOv8n (Nano)
- **Input:** 640×640 (auto-resized)
- **Training:** 500 epochs on Roboflow dataset
- **Class:** Single class ("swellings")
- **Default confidence:** 0.02
- **File:** `weights.pt` (22.5 MB)

---

## 🧪 Testing

```bash
# Run with sample images
ascam pipeline \
  --input test_images/ \
  --output test_results/ \
  --classifier models/best_model.keras \
  --detector models/weights.pt

# Check results
ls test_results/
cat test_results/results.json
```

Expected output: ~173 total swellings across 12 test images.

---

## 📝 Legacy Notebooks

The original Jupyter notebooks are preserved in:
- `Classification_Model/` - Training notebook
- `Object_Detection_Model/` - Training and inference notebooks
- `Running_Code/` - Complete pipeline notebook

These notebooks are designed for Google Colab and require Google Drive mounting.

## Contributing

Thank you for wanting to improve ASCAM! If you encounter any bugs, have questions, or would like to contribute new features, feel free to open an issue or submit a pull request.

