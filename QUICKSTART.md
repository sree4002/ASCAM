# ASCAM Quick Start Guide

Get started with ASCAM in 5 minutes!

## 1. Installation

```bash
# Clone the repository
git clone https://github.com/sree4002/ASCAM.git
cd ASCAM

# Install dependencies
pip install -e .

# This will install:
# - PyTorch, timm (for EfficientNet-B0 classification)
# - Ultralytics YOLO (for detection)
# - OpenCV, NumPy, Pillow, and other dependencies
```

## 2. Model Weights

Model files should be in the `models/` directory:
- `models/classifier.pt` -- EfficientNet-B0 classifier (PyTorch)
- `models/yolov8s_best.pt` -- YOLOv8s detector

## 3. Run Your First Analysis

### Option A: Full Pipeline (Recommended)

```bash
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --classifier models/classifier.pt \
  --detector models/yolov8s_best.pt
```

### Option B: Detection Only (Faster)

If you know your images contain swellings, skip classification:

```bash
ascam detect \
  --input test_images/ \
  --output results/ \
  --model models/yolov8s_best.pt
```

### Option C: Python API

```python
from ascam import ASCAMPipeline

# Initialize pipeline
pipeline = ASCAMPipeline(
    classifier_path="models/classifier.pt",
    detector_path="models/yolov8s_best.pt"
)

# Process images
results = pipeline.process_directory(
    input_dir="test_images/",
    output_dir="results/"
)

# Print summary
print(f"Total swellings: {sum(r.count for r in results)}")
```

## 4. View Results

After processing:
```bash
# View annotated images
ls results/*.JPG

# View detection statistics
cat results/results.json

# Or in CSV format
ascam pipeline --input test_images/ --output results/ --format csv
cat results/results.csv
```

## 5. Customize Settings

Generate a config file and edit it:

```bash
ascam config --output my_config.yaml
# Edit my_config.yaml with your preferred settings
ascam pipeline --input test_images/ --output results/ --config my_config.yaml
```

## Common Options

```bash
# Adjust detection confidence threshold
ascam pipeline ... --conf 0.25

# Skip classification (faster, less filtering)
ascam pipeline ... --skip-classify

# Save as CSV instead of JSON
ascam pipeline ... --format csv

# Show confidence scores on boxes
ascam detect ... --show-conf

# Verbose output for debugging
ascam pipeline ... --verbose
```

## Training

```bash
# Train CNN classifier (EfficientNet-B0 with Focal Loss)
ascam train-cnn --data data/classifier/ --epochs 50 --output models/

# Train YOLO detector (paper-compliant parameters)
ascam train-yolo --data data/detection/data.yaml --epochs 500 --output models/
```

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'torch'`
**Solution:** Run `pip install -e .` again

**Problem:** Model files not found
**Solution:** Ensure `models/classifier.pt` and `models/yolov8s_best.pt` exist

**Problem:** Out of memory
**Solution:** Process images in smaller batches or use a machine with more RAM

**Problem:** Slow inference
**Solution:** Use a GPU-enabled machine or skip classification with `--skip-classify`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Run `bash scripts/reproduce.sh` for full reproducibility
- Adjust detection thresholds in config file for your specific use case
- Export results to CSV for analysis in Excel/R/Python

## Support

For issues or questions:
- Open an issue: https://github.com/sree4002/ASCAM/issues
- Check documentation: [README.md](README.md)
