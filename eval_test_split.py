"""Evaluate classifier on the held-out 45-image test split only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ascam.models.classifier import SwellingClassifier
from ascam.training.train_cnn import SwellingDataset, get_transforms

DATA_DIR = "data/classifier"
MODEL_PATH = "models/classifier.pt"


def main():
    # 1. Reconstruct exact test split (same as training)
    full_dataset = SwellingDataset(DATA_DIR, transform=None)
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    generator = torch.Generator().manual_seed(42)
    _, _, test_idx = torch.utils.data.random_split(
        range(n), [n_train, n_val, n_test], generator=generator
    )

    data_path = Path(DATA_DIR)
    test_files = {
        str(full_dataset.images[i].relative_to(data_path))
        for i in test_idx.indices
    }
    val_transform = get_transforms(is_train=False)
    test_dataset = SwellingDataset(DATA_DIR, transform=val_transform, split_files=test_files)

    print(f"Test split: {len(test_dataset)} images")
    print(f"  Positive: {sum(test_dataset.labels)}")
    print(f"  Negative: {len(test_dataset.labels) - sum(test_dataset.labels)}")

    # 2. Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = SwellingClassifier(
        model_path=MODEL_PATH, image_size=(224, 224), device=str(device)
    )

    # 3. Run inference on test split
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = classifier.model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_pred = y_probs.argmax(axis=1)
    pos_probs = y_probs[:, 1]  # probability of swelling class

    # 4. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, pos_probs)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    # 5. Print results
    print("\n" + "=" * 60)
    print("CLASSIFIER TEST SPLIT EVALUATION (45 images)")
    print("=" * 60)
    print(f"  Accuracy:   {acc * 100:.1f}%")
    print(f"  Precision:  {prec * 100:.1f}%")
    print(f"  Recall:     {rec * 100:.1f}%")
    print(f"  F1 Score:   {f1 * 100:.1f}%")
    print(f"  AUC-ROC:    {auc_roc:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP={tp}  FP={fp}")
    print(f"  FN={fn}  TN={tn}")
    print()
    print(classification_report(y_true, y_pred, target_names=["No Swelling", "Swelling"]))


if __name__ == "__main__":
    main()
