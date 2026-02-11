#!/bin/bash
set -e

# ASCAM Reproducibility Script
# ============================
# One-command end-to-end pipeline for reviewers.
# Requires training data in data/classifier/ and data/detection/

echo "=== ASCAM Reproducibility Pipeline ==="
echo ""

# 1. Train CNN classifier (EfficientNet-B0)
echo "=== Stage 1: Training CNN Classifier ==="
ascam train-cnn --data data/classifier/ --epochs 50 --output models/

# 2. Train YOLO detector (paper-compliant)
echo "=== Stage 2: Training YOLO Detector ==="
ascam train-yolo --data data/detection/data.yaml --epochs 500 --output models/

# 3. Evaluate thresholds
echo "=== Stage 3: Evaluating Detection Thresholds ==="
python -m ascam.training.evaluate \
  --classifier models/classifier.pt \
  --detector models/yolov8s_best.pt \
  --data data/detection/test/ \
  --sweep-conf 0.01 0.50 0.01

# 4. Run full pipeline on test images
echo "=== Stage 4: Running Full Pipeline ==="
ascam pipeline \
  --input test_images/ \
  --output results/ \
  --classifier models/classifier.pt \
  --detector models/yolov8s_best.pt

echo "=== Done. Results in results/ ==="
