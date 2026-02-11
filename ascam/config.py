"""Configuration management for ASCAM."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "classifier": {
        "model_path": "models/classifier.pt",
        "image_size": [224, 224],
        "threshold": 0.5,
        "model_name": "efficientnet_b0",
        "num_classes": 2,
    },
    "detector": {
        "model_path": "models/yolov8s_best.pt",
        "conf_threshold": 0.25,
        "iou_threshold": 0.50,
        "max_detections": 1000,
    },
    "visualization": {
        "box_color": [0, 0, 255],  # BGR: Red
        "box_thickness": 10,
        "show_count": True,
        "show_confidence": False,
    },
    "pipeline": {
        "skip_classification": False,
        "save_results": True,
        "results_format": "json",
    },
    "training": {
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "warmup_epochs": 5,
        "patience": 10,
        "use_focal_loss": True,
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "label_smoothing": 0.1,
        "mixed_precision": True,
        "dropout": 0.3,
        "seed": 42,
    },
    "augmentation": {
        "rotation_limit": 45,
        "horizontal_flip": True,
        "vertical_flip": False,
        "brightness_limit": 0.2,
        "contrast_limit": 0.2,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    },
}


class Config:
    """Configuration manager for ASCAM."""

    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.

        Args:
            config_dict: Configuration dictionary (uses defaults if None)
        """
        self.config = config_dict or DEFAULT_CONFIG.copy()

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(f"Config file {yaml_path} not found, using defaults")
            return cls()

        try:
            with open(yaml_path, "r") as f:
                config_dict = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {yaml_path}")
            return cls(config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {yaml_path}: {e}")
            return cls()

    def save_yaml(self, yaml_path: Union[str, Path]):
        """
        Save configuration to YAML file.

        Args:
            yaml_path: Path to save YAML file
        """
        yaml_path = Path(yaml_path)
        try:
            with open(yaml_path, "w") as f:
                yaml.dump(self.config, f, default_flow_style=False)
            logger.info(f"Saved configuration to {yaml_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {yaml_path}: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (supports dot notation, e.g., 'classifier.threshold')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        keys = key.split(".")
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary.

        Args:
            updates: Dictionary with updates
        """

        def deep_update(base: Dict, updates: Dict) -> Dict:
            for key, value in updates.items():
                if isinstance(value, dict) and key in base:
                    base[key] = deep_update(base[key], value)
                else:
                    base[key] = value
            return base

        self.config = deep_update(self.config, updates)

    def __repr__(self) -> str:
        """String representation."""
        return f"Config({self.config})"


def create_default_config(output_path: Union[str, Path]):
    """
    Create default configuration file.

    Args:
        output_path: Path to save default config
    """
    config = Config()
    config.save_yaml(output_path)
    logger.info(f"Created default configuration at {output_path}")
