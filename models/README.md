# Model Files

This directory contains the trained model weights for ASCAM.

## Required Files

1. **classifier.pt** (16 MB)
   - EfficientNet-B0 binary classification model
   - Detects presence/absence of axonal swellings
   - Built with PyTorch + timm

2. **yolov8s_best.pt** (23 MB)
   - YOLOv8s object detection model
   - Localizes individual axonal swellings
   - Built with Ultralytics YOLO

## Model Information

### Classification Model (classifier.pt)
- Input size: 224x224 RGB images
- Architecture: EfficientNet-B0 (transfer learning)
- Output: 2-class softmax (no_swelling, swelling)
- Training: Focal Loss, AdamW, OneCycleLR

### Detection Model (yolov8s_best.pt)
- Input size: 1280x1280
- Architecture: YOLOv8s
- Inference: Test-time augmentation (TTA) enabled
- Training: 500 epochs, mosaic disabled, AdamW
- Class: "swelling" (single class)
- Default thresholds: conf=0.25, iou=0.50
