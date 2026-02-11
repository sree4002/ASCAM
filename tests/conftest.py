"""Shared fixtures for ASCAM tests."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    path = temp_dir / "test_image.jpg"
    img.save(str(path))
    return path


@pytest.fixture
def sample_image_dir(temp_dir):
    """Create a directory with sample test images."""
    image_dir = temp_dir / "images"
    image_dir.mkdir()

    for i in range(3):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(str(image_dir / f"test_{i}.jpg"))

    # Add different extensions
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(str(image_dir / "test.png"))

    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img.save(str(image_dir / "test.bmp"))

    return image_dir


@pytest.fixture
def classifier_data_dir(temp_dir):
    """Create a classifier training data directory."""
    data_dir = temp_dir / "classifier_data"
    swelling_dir = data_dir / "swelling"
    no_swelling_dir = data_dir / "no_swelling"
    swelling_dir.mkdir(parents=True)
    no_swelling_dir.mkdir(parents=True)

    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(str(swelling_dir / f"pos_{i}.jpg"))

    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img.save(str(no_swelling_dir / f"neg_{i}.jpg"))

    return data_dir
