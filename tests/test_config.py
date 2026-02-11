"""Tests for configuration management."""

import pytest
import yaml
from pathlib import Path

from ascam.config import Config, DEFAULT_CONFIG, create_default_config


class TestConfig:
    """Tests for Config class."""

    def test_default_config(self):
        config = Config()
        assert config.get('classifier.model_path') == 'models/classifier.pt'
        assert config.get('classifier.image_size') == [224, 224]
        assert config.get('classifier.model_name') == 'efficientnet_b0'
        assert config.get('classifier.num_classes') == 2
        assert config.get('detector.conf_threshold') == 0.25
        assert config.get('detector.iou_threshold') == 0.50

    def test_get_dot_notation(self):
        config = Config()
        assert config.get('classifier.threshold') == 0.5
        assert config.get('detector.max_detections') == 1000
        assert config.get('visualization.show_count') is True

    def test_get_default_value(self):
        config = Config()
        assert config.get('nonexistent.key', 'default') == 'default'
        assert config.get('classifier.nonexistent', 42) == 42

    def test_set_dot_notation(self):
        config = Config()
        config.set('classifier.threshold', 0.7)
        assert config.get('classifier.threshold') == 0.7

    def test_set_creates_nested(self):
        config = Config()
        config.set('new.nested.key', 'value')
        assert config.get('new.nested.key') == 'value'

    def test_update(self):
        config = Config()
        config.update({'classifier': {'threshold': 0.9}})
        assert config.get('classifier.threshold') == 0.9
        # Other classifier keys should remain
        assert config.get('classifier.model_path') == 'models/classifier.pt'

    def test_training_section_exists(self):
        config = Config()
        assert config.get('training.epochs') == 50
        assert config.get('training.use_focal_loss') is True
        assert config.get('training.focal_alpha') == 0.25

    def test_augmentation_section_exists(self):
        config = Config()
        assert config.get('augmentation.normalize_mean') == [0.485, 0.456, 0.406]
        assert config.get('augmentation.normalize_std') == [0.229, 0.224, 0.225]

    def test_save_and_load_yaml(self, temp_dir):
        config = Config()
        yaml_path = temp_dir / 'test_config.yaml'
        config.save_yaml(yaml_path)

        loaded = Config.from_yaml(yaml_path)
        assert loaded.get('classifier.model_path') == config.get('classifier.model_path')
        assert loaded.get('detector.conf_threshold') == config.get('detector.conf_threshold')

    def test_from_yaml_missing_file(self):
        config = Config.from_yaml('nonexistent.yaml')
        # Should return default config
        assert config.get('classifier.model_path') == 'models/classifier.pt'

    def test_create_default_config(self, temp_dir):
        output_path = temp_dir / 'default.yaml'
        create_default_config(output_path)
        assert output_path.exists()

        with open(output_path) as f:
            data = yaml.safe_load(f)
        assert 'classifier' in data
        assert 'detector' in data

    def test_repr(self):
        config = Config()
        repr_str = repr(config)
        assert 'Config(' in repr_str
