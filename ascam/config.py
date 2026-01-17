"""Configuration management for ASCAM."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


DEFAULT_CONFIG = {
    "classifier": {
        "model_path": "models/best_model.keras",
        "image_size": [200, 200],
        "threshold": 0.5
    },
    "detector": {
        "model_path": "models/weights.pt",
        "conf_threshold": 0.02,
        "iou_threshold": 0.30,
        "max_detections": 1000
    },
    "visualization": {
        "box_color": [0, 0, 255],  # BGR: Red
        "box_thickness": 10,
        "show_count": True,
        "show_confidence": False
    },
    "pipeline": {
        "skip_classification": False,
        "save_results": True,
        "results_format": "json"
    }
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
    def from_yaml(cls, yaml_path: Union[str, Path]) -> 'Config':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            Config object
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            logger.warning(
                f"Config file {yaml_path} not found, using defaults"
            )
            return cls()

        try:
            with open(yaml_path, 'r') as f:
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
            with open(yaml_path, 'w') as f:
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
        keys = key.split('.')
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
        keys = key.split('.')
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
