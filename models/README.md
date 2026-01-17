# Model Files

This directory contains the trained model weights for ASCAM.

## Required Files

1. **best_model.keras** (162 MB)
   - Binary classification model
   - Detects presence/absence of axonal swellings
   - Built with TensorFlow/Keras

2. **weights.pt** (22.5 MB)
   - YOLOv8 object detection model
   - Localizes individual axonal swellings
   - Built with Ultralytics YOLO

## Download Instructions

If these files are not already present, you can find them in the repository:
- Location in repo: `Running_Code/best_model.keras` and `Running_Code/weights.pt`
- Or download from Git LFS if configured

## Model Information

### Classification Model
- Input size: 200x200 RGB images
- Architecture: CNN with batch normalization
- Output: Binary (swelling/no swelling)
- Training epochs: 100 (with early stopping)

### Detection Model
- Input size: 640x640 (auto-resized)
- Architecture: YOLOv8n
- Training epochs: 500
- Class: "swellings" (single class)
- Confidence threshold: 0.02 (default)
