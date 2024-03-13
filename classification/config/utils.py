import yaml
from typing import Dict, Any

from classification.config.schemas import ExperimentConfig


def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_config(config_path: str, overwrite_dict=None) -> Dict[Any, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if overwrite_dict:
        nested_update(config, overwrite_dict)
    return ExperimentConfig.from_dict(config)
