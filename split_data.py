import os
import shutil
import random
from pathlib import Path

source_folder = "/Users/sreenidhi_surineni/Downloads/project_153242_dataset_2024_07_13_10_46_10_yolo 1.1/obj_train_data"
source_path = Path(source_folder)

output_base = Path("data/detection")
splits = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for split in splits:
    (output_base / split / 'images').mkdir(parents=True, exist_ok=True)
    (output_base / split / 'labels').mkdir(parents=True, exist_ok=True)

image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
images_with_labels = []

for img_path in source_path.iterdir():
    if img_path.suffix in image_extensions:
        label_path = source_path / (img_path.stem + '.txt')
        if label_path.exists():
            images_with_labels.append((img_path, label_path))

print(f"Found {len(images_with_labels)} images with labels")

random.seed(42)
random.shuffle(images_with_labels)

n = len(images_with_labels)
train_end = int(n * splits['train'])
val_end = train_end + int(n * splits['val'])

split_assignments = {
    'train': images_with_labels[:train_end],
    'val': images_with_labels[train_end:val_end],
    'test': images_with_labels[val_end:]
}

for split, files in split_assignments.items():
    for img_path, label_path in files:
        shutil.copy2(img_path, output_base / split / 'images' / img_path.name)
        shutil.copy2(label_path, output_base / split / 'labels' / label_path.name)
    print(f"{split}: {len(files)} images")

yaml_content = """path: data/detection
train: train/images
val: val/images
test: test/images
nc: 1
names: ['swelling']
"""

with open(output_base / 'data.yaml', 'w') as f:
    f.write(yaml_content)

print(f"Done! data.yaml created")
