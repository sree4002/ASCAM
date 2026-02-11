"""Tests for image utility functions."""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from ascam.utils.image_utils import load_image, preprocess_image, save_image, get_image_files


class TestLoadImage:
    """Tests for load_image."""

    def test_load_valid_image(self, sample_image):
        image = load_image(sample_image)
        assert image is not None
        assert isinstance(image, np.ndarray)
        assert len(image.shape) == 3

    def test_load_nonexistent_image(self):
        image = load_image("nonexistent.jpg")
        assert image is None

    def test_load_invalid_path(self):
        image = load_image("")
        assert image is None


class TestPreprocessImage:
    """Tests for preprocess_image."""

    def test_resize(self, sample_image):
        image = load_image(sample_image)
        processed = preprocess_image(image, target_size=(100, 100), normalize=False, to_rgb=False)
        assert processed.shape[:2] == (100, 100)

    def test_normalize(self, sample_image):
        image = load_image(sample_image)
        processed = preprocess_image(image, normalize=True, to_rgb=False)
        assert processed.max() <= 1.0
        assert processed.min() >= 0.0
        assert processed.dtype == np.float32

    def test_to_rgb(self, sample_image):
        image = load_image(sample_image)
        processed = preprocess_image(image, target_size=(100, 100), normalize=False, to_rgb=True)
        assert processed.shape == (100, 100, 3)


class TestSaveImage:
    """Tests for save_image."""

    def test_save_image(self, temp_dir):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output_path = temp_dir / "saved.jpg"
        result = save_image(image, output_path)
        assert result is True
        assert output_path.exists()

    def test_save_to_nonexistent_dir(self, temp_dir):
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        output_path = temp_dir / "nonexistent_dir" / "saved.jpg"
        result = save_image(image, output_path)
        assert result is False


class TestGetImageFiles:
    """Tests for get_image_files."""

    def test_find_images(self, sample_image_dir):
        files = get_image_files(sample_image_dir)
        assert len(files) == 5  # 3 jpg + 1 png + 1 bmp
        assert all(isinstance(f, Path) for f in files)

    def test_case_insensitive(self, temp_dir):
        image_dir = temp_dir / "case_test"
        image_dir.mkdir()

        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        img.save(str(image_dir / "test.JPG"))
        img.save(str(image_dir / "test2.jpg"))
        img.save(str(image_dir / "test3.PNG"))

        files = get_image_files(image_dir)
        assert len(files) == 3

    def test_nonexistent_directory(self):
        files = get_image_files("nonexistent_dir")
        assert files == []

    def test_empty_directory(self, temp_dir):
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        files = get_image_files(empty_dir)
        assert files == []

    def test_sorted_output(self, sample_image_dir):
        files = get_image_files(sample_image_dir)
        names = [f.name for f in files]
        assert names == sorted(names)

    def test_tiff_support(self, temp_dir):
        image_dir = temp_dir / "tiff_test"
        image_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8))
        img.save(str(image_dir / "test.tiff"))
        img.save(str(image_dir / "test2.tif"))

        files = get_image_files(image_dir)
        assert len(files) == 2
