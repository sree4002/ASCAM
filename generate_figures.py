"""Generate classification model figures: training curves, confusion matrix, ROC curve."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from sklearn.metrics import confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ascam.models.classifier import SwellingClassifier
from ascam.training.train_cnn import SwellingDataset, get_transforms

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = "data/classifier"
MODEL_PATH = "models/classifier.pt"
RESULTS_PATH = "models/training_results.json"


# ── 1. Training Curves ──────────────────────────────────────────────────────

def plot_training_curves(results_path: str, save_path: Path):
    with open(results_path) as f:
        results = json.load(f)

    history = results["history"]
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    ax1.plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#2196F3", markersize=4)
    ax1.plot(epochs, history["val_loss"], "s-", label="Val Loss", color="#FF5722", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc = [a * 100 for a in history["val_acc"]]
    ax2.plot(epochs, train_acc, "o-", label="Train Acc", color="#2196F3", markersize=4)
    ax2.plot(epochs, val_acc, "s-", label="Val Acc", color="#FF5722", markersize=4)

    best_epoch = int(np.argmax(history["val_acc"])) + 1
    best_val = max(val_acc)
    ax2.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5, label=f"Best epoch ({best_epoch})")
    ax2.annotate(f"{best_val:.1f}%", xy=(best_epoch, best_val),
                 xytext=(best_epoch + 1, best_val - 5),
                 arrowprops=dict(arrowstyle="->", color="gray"),
                 fontsize=10, color="#FF5722")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle("EfficientNet-B0 Classification Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


# ── 2 & 3. Confusion Matrix + ROC (need test set predictions) ───────────────

def reconstruct_test_set():
    """Recreate the exact test split using the same seed as training."""
    full_dataset = SwellingDataset(DATA_DIR, transform=None)
    n = len(full_dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    generator = torch.Generator().manual_seed(42)
    _, _, test_idx = torch.utils.data.random_split(
        range(n), [n_train, n_val, n - n_train - n_val], generator=generator
    )

    data_path = Path(DATA_DIR)
    test_files = {
        str(full_dataset.images[i].relative_to(data_path))
        for i in test_idx.indices
    }
    val_transform = get_transforms(is_train=False)
    test_dataset = SwellingDataset(DATA_DIR, transform=val_transform, split_files=test_files)
    return test_dataset


def get_test_predictions(test_dataset: SwellingDataset):
    """Run model on test set, return true labels, predicted labels, and probabilities."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classifier = SwellingClassifier(
        model_path=MODEL_PATH,
        image_size=(224, 224),
        device=str(device),
    )

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

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)

    return all_labels, all_preds, all_probs


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Swelling", "Swelling"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            current_text = ax.texts[i * 2 + j]
            current_text.set_text(f"{cm[i, j]}\n({pct:.1f}%)")

    acc = np.trace(cm) / total * 100
    ax.set_title(f"Confusion Matrix on Test Set\nAccuracy: {acc:.1f}%",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path: Path):
    # Probability of positive class (swelling = class 1)
    pos_probs = y_probs[:, 1]

    fpr, tpr, thresholds = roc_curve(y_true, pos_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

    # Mark the operating point closest to threshold=0.5
    idx_05 = np.argmin(np.abs(thresholds - 0.5))
    ax.plot(fpr[idx_05], tpr[idx_05], "ro", markersize=8, label=f"Threshold=0.5 (FPR={fpr[idx_05]:.2f}, TPR={tpr[idx_05]:.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Swelling Classification", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    print(f"  AUC: {roc_auc:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("Generating classification figures")
    print("=" * 50)

    # Figure 1: Training curves
    print("\n[1/3] Training curves...")
    plot_training_curves(RESULTS_PATH, OUTPUT_DIR / "training_curves.png")

    # Reconstruct test set and get predictions
    print("\n[2/3] Confusion matrix (loading test set)...")
    test_dataset = reconstruct_test_set()
    y_true, y_pred, y_probs = get_test_predictions(test_dataset)
    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR / "confusion_matrix.png")

    # Figure 3: ROC curve
    print("\n[3/3] ROC curve...")
    plot_roc_curve(y_true, y_probs, OUTPUT_DIR / "roc_curve.png")

    print("\nDone! All figures saved to outputs/")
