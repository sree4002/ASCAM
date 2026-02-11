"""
ASCAM CNN Training
==================
Train EfficientNet-B0 for swelling vs no_swelling classification.

Features:
- Focal Loss for class imbalance
- OneCycleLR scheduler with warmup
- Mixed precision training
- Early stopping with patience
- Checkpoint management
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

import timm

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    from torchvision import transforms

    ALBUMENTATIONS_AVAILABLE = False

from ascam.models.classifier import FocalLoss

logger = logging.getLogger(__name__)


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Create EfficientNet-B0 model for classification."""
    model = timm.create_model(
        model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout
    )
    logger.info(
        f"Created {model_name} (pretrained={pretrained}, classes={num_classes})"
    )
    return model


class SwellingDataset(Dataset):
    """Dataset for swelling classification with flexible folder naming."""

    POSITIVE_NAMES = {"swelling", "swellings", "positive", "pos"}
    NEGATIVE_NAMES = {
        "no_swelling",
        "no swelling",
        "no_swellings",
        "negative",
        "neg",
        "non_swelling",
        "non-swelling",
    }

    def __init__(self, data_dir: str, transform=None, split_files=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []

        # Find positive and negative folders
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
            name_lower = subdir.name.lower().strip()
            if name_lower in self.POSITIVE_NAMES:
                label = 1
            elif name_lower in self.NEGATIVE_NAMES:
                label = 0
            else:
                continue

            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            files = [
                f
                for f in subdir.iterdir()
                if f.is_file() and f.suffix.lower() in extensions
            ]

            if split_files is not None:
                files = [f for f in files if f.name in split_files]

            self.images.extend(files)
            self.labels.extend([label] * len(files))

        logger.info(
            f"Dataset: {len(self.images)} images "
            f"({sum(self.labels)} positive, {len(self.labels) - sum(self.labels)} negative)"
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(str(self.images[idx])).convert("RGB")
        image_np = np.array(image)
        label = self.labels[idx]

        if self.transform is not None:
            if ALBUMENTATIONS_AVAILABLE and isinstance(self.transform, A.Compose):
                transformed = self.transform(image=image_np)
                image_tensor = transformed["image"]
            else:
                image_tensor = self.transform(image)
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        return image_tensor, label


def get_transforms(image_size=(224, 224), is_train=False):
    """Get data transforms for training or inference."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if ALBUMENTATIONS_AVAILABLE:
        if is_train:
            return A.Compose(
                [
                    A.Resize(image_size[0], image_size[1]),
                    A.HorizontalFlip(p=0.5),
                    A.Rotate(limit=45, p=0.5),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2, contrast_limit=0.2, p=0.5
                    ),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(image_size[0], image_size[1]),
                    A.Normalize(mean=mean, std=std),
                    ToTensorV2(),
                ]
            )
    else:
        if is_train:
            return transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(45),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )


def compute_class_weights(dataset: SwellingDataset) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    labels = np.array(dataset.labels)
    counts = np.bincount(labels)
    weights = len(labels) / (len(counts) * counts.astype(float))
    return torch.FloatTensor(weights)


class Trainer:
    """CNN Trainer with mixed precision, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        data_dir: str,
        output_dir: str = "models/",
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.model = model
        self.epochs = epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        logger.info(f"Using device: {self.device}")

        # Create datasets and dataloaders
        train_transform = get_transforms(is_train=True)
        val_transform = get_transforms(is_train=False)

        full_dataset = SwellingDataset(data_dir, transform=None)
        n = len(full_dataset)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        n_test = n - n_train - n_val

        generator = torch.Generator().manual_seed(42)
        train_idx, val_idx, test_idx = torch.utils.data.random_split(
            range(n), [n_train, n_val, n_test], generator=generator
        )

        # Create split-specific datasets
        train_files = {full_dataset.images[i].name for i in train_idx.indices}
        val_files = {full_dataset.images[i].name for i in val_idx.indices}
        test_files = {full_dataset.images[i].name for i in test_idx.indices}

        train_dataset = SwellingDataset(
            data_dir, transform=train_transform, split_files=train_files
        )
        val_dataset = SwellingDataset(
            data_dir, transform=val_transform, split_files=val_files
        )
        test_dataset = SwellingDataset(
            data_dir, transform=val_transform, split_files=test_files
        )

        self.dataloaders = {
            "train": DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            ),
            "test": DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            ),
        }

        # Loss function
        if use_focal_loss:
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
        else:
            class_weights = compute_class_weights(train_dataset)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            logger.info("Using CrossEntropyLoss with class weights")

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        # Scheduler
        steps_per_epoch = len(self.dataloaders["train"])
        total_steps = steps_per_epoch * epochs
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=5 / epochs,
            anneal_strategy="cos",
        )

        # Mixed precision
        self.use_amp = torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Early stopping
        self.best_val_acc = 0.0
        self.patience = patience
        self.patience_counter = 0

        # History
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.dataloaders["train"], desc="Training", leave=False)

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100.*correct/total:.2f}%"}
            )

        return running_loss / total, correct / total

    @torch.no_grad()
    def validate(self, split: str = "val") -> Tuple[float, float]:
        """Validate on val or test set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.dataloaders[split]:
            images = images.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return running_loss / total, correct / total

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
        }

        torch.save(checkpoint, self.output_dir / "checkpoint_latest.pt")

        if is_best:
            torch.save(checkpoint, self.output_dir / "checkpoint_best.pt")
            torch.save(self.model.state_dict(), self.output_dir / "classifier.pt")

    def train(self) -> Dict:
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("STARTING CNN TRAINING")
        logger.info("=" * 60)

        start_time = time.time()
        epoch = 0

        for epoch in range(1, self.epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate("val")
            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            logger.info(
                f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%"
            )
            logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                logger.info(f"  New best model! (Val Acc: {val_acc*100:.2f}%)")
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            if self.patience_counter >= self.patience:
                logger.info(f"\nEarly stopping triggered (patience={self.patience})")
                break

        # Final test evaluation
        best_weights = self.output_dir / "classifier.pt"
        if best_weights.exists():
            self.model.load_state_dict(
                torch.load(best_weights, map_location=self.device)
            )
        test_loss, test_acc = self.validate("test")

        logger.info(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

        total_time = time.time() - start_time

        results = {
            "best_val_acc": self.best_val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "epochs_trained": epoch,
            "training_time_seconds": total_time,
            "history": self.history,
        }

        with open(self.output_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
        logger.info(f"  Best Val Acc: {self.best_val_acc*100:.2f}%")
        logger.info(f"  Test Acc:     {test_acc*100:.2f}%")
        logger.info(f"  Model saved:  {self.output_dir / 'classifier.pt'}")

        return results
