"""
Configuration module initialization.
Import helper functions from here to access configs from anywhere in your project.
"""

from .config_manager import ConfigManager, get_config, get_config_value, get_main_config

# Initialize the config manager once when the module is imported for the first time.
config_manager = ConfigManager()

# Export commonly used functions for easy access
__all__ = ['config_manager', 'get_config', 'get_config_value', 'get_main_config']