import yaml
from typing import Dict, Any

from src.config.schemas import ExperimentConfig


def parse_config(config_path: str) -> Dict[Any, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return ExperimentConfig.from_dict(config)
