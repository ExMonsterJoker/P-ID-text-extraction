import yaml
from pathlib import Path


class ConfigLoader:
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.configs = {}

    def load_configs(self):
        config_files = [
            "base.yaml",
            "text_detection.yaml",
            "postprocessing.yaml",
            "grouping.yaml",
            "ocr.yaml"
        ]

        for file in config_files:
            with open(self.config_dir / file) as f:
                config_name = file.split(".")[0]
                self.configs[config_name] = yaml.safe_load(f)

        return self.configs

    def get_config(self, module_name):
        return self.configs.get(module_name, {})