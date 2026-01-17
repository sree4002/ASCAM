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
# - TensorFlow (for classification)
# - Ultralytics YOLO (for detection)
# - OpenCV, NumPy, and other dependencies
```

## 2. Download Model Weights

Model files are located in the `Running_Code/` directory:
- `Running_Code/best_model.keras` → Copy to `models/best_model.keras`
- `Running_Code/weights.pt` → Copy to `models/weights.pt`

```bash
# Create models directory and copy weights
mkdir -p models
cp Running_Code/best_model.keras models/
cp Running_Code/weights.pt models/
```

## 3. Run Your First Analysis

### Option A: Full Pipeline (Recommended)

```bash
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --classifier models/best_model.keras \
  --detector models/weights.pt
```

### Option B: Detection Only (Faster)

If you know your images contain swellings, skip classification:

```bash
ascam detect \
  --input test_images/ \
  --output results/ \
  --model models/weights.pt
```

### Option C: Python API

```python
from ascam import ASCAMPipeline

# Initialize pipeline
pipeline = ASCAMPipeline(
    classifier_path="models/best_model.keras",
    detector_path="models/weights.pt"
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
# Adjust detection confidence threshold (lower = more detections)
ascam pipeline ... --conf 0.01

# Skip classification (faster, less filtering)
ascam pipeline ... --skip-classify

# Save as CSV instead of JSON
ascam pipeline ... --format csv

# Show confidence scores on boxes
ascam detect ... --show-conf

# Verbose output for debugging
ascam pipeline ... --verbose
```

## Troubleshooting

**Problem:** `ModuleNotFoundError: No module named 'tensorflow'`
**Solution:** Run `pip install -e .` again

**Problem:** Model files not found
**Solution:** Copy model files from `Running_Code/` to `models/` directory

**Problem:** Out of memory
**Solution:** Process images in smaller batches or use a machine with more RAM

**Problem:** Slow inference
**Solution:** Use a GPU-enabled machine or skip classification with `--skip-classify`

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check example notebooks in `Running_Code/` for Colab usage
- Adjust detection thresholds in config file for your specific use case
- Export results to CSV for analysis in Excel/R/Python

## Support

For issues or questions:
- Open an issue: https://github.com/sree4002/ASCAM/issues
- Check documentation: [README.md](README.md)
