import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from functools import lru_cache


class ConfigManager:
    """
    Centralized configuration manager using a singleton pattern.
    Handles loading and accessing configuration settings from a YAML file.
    This ensures the configuration is loaded only once during the application's lifetime.
    """
    _instance: Optional['ConfigManager'] = None
    _initialized: bool = False

    def __new__(cls):
        """Ensure only one instance of ConfigManager exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the ConfigManager by finding the project root, locating,
        and loading the configuration file. Prevents re-initialization.
        """
        if self._initialized:
            return

        self.project_root: Path = self._find_project_root()
        self.config_path: Path = self.project_root / "configs" / "base.yaml"
        self.config: Dict[str, Any] = self._load_config_file()

        if not self.config:
            logging.warning("Configuration is empty. Please check config file content.")

        self._initialized = True
        logging.info(f"ConfigManager initialized. Loaded config from: {self.config_path}")

    def _find_project_root(self) -> Path:
        """
        Finds the project root directory.
        It assumes this script is in a subdirectory of the project root (e.g., 'configs').
        """
        # Path(__file__) is the path to this file (config_manager.py)
        # .resolve() makes it an absolute path
        # .parent is the 'configs' directory
        # .parent is the project root
        return Path(__file__).resolve().parent.parent

    def _load_config_file(self) -> Dict[str, Any]:
        """Loads the main YAML configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logging.error(f"Config file not found at: {self.config_path}")
            return {}
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file {self.config_path}: {e}")
            return {}

    @lru_cache(maxsize=128)
    def get_value(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the configuration using a dot-separated key.
        Example: get_value('pipeline.input_dir')

        Args:
            key (str): The dot-separated key for the desired value.
            default (Any, optional): The value to return if the key is not found.

        Returns:
            Any: The configuration value or the default.
        """
        keys = key.split('.')
        value = self.config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logging.warning(f"Config key '{key}' not found. Returning default: {default}")
            return default

    def get_section(self, section_key: str) -> Dict[str, Any]:
        """
        Retrieves an entire section of the configuration.

        Args:
            section_key (str): The key for the desired section (e.g., 'ocr').

        Returns:
            Dict[str, Any]: The configuration section as a dictionary, or an empty dict.
        """
        return self.config.get(section_key, {})

    def get_full_config(self) -> Dict[str, Any]:
        """Returns the entire loaded configuration dictionary."""
        return self.config


# --- Helper Functions for Easy Access ---

# Initialize the config manager once.
# This instance will be shared across the entire application.
_config_manager_instance = ConfigManager()


def get_main_config() -> Dict[str, Any]:
    """Convenience function to get the entire configuration dictionary."""
    return _config_manager_instance.get_full_config()


def get_config(section_key: str) -> Dict[str, Any]:
    """Convenience function to get a specific section from the config."""
    return _config_manager_instance.get_section(section_key)


def get_config_value(key: str, default: Any = None) -> Any:
    """Convenience function to get a specific value from the config using a dot-separated key."""
    return _config_manager_instance.get_value(key, default)


# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    print("\n--- Testing ConfigManager ---")

    # Accessing values using helper functions
    print(f"Project Root: {_config_manager_instance.project_root}")

    device = get_config_value('ocr.gpu')
    print(f"Pipeline Device: {device}")

    tile_size = get_config_value('grouping.IOU_threshold')
    print(f"SAHI Tile Size: {tile_size}")


    # Getting a whole section
    ocr_config = get_config('ocr')
    print("\nOCR Config Section:")
    print(ocr_config)