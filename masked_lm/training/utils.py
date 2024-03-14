import argparse
from typing import Tuple, Any

from masked_lm.config.utils import parse_config
from masked_lm.data.data_loader import RottenDataLoaderForMaskedLM
from masked_lm.models.lightning_wrapper import (
    RottenTomatoesDomainAdaptationModel,
    LightingModelWrapperForMaskedLM,
)
from masked_lm.utils.factories import optimizer_factory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to .yaml file that contains configuration details",
    )
    return parser.parse_args()


def setup(config_path: str, overwrite_dict=None) -> Tuple[Any]:
    config = parse_config(config_path, overwrite_dict)
    data = RottenDataLoaderForMaskedLM(config.data)
    model = RottenTomatoesDomainAdaptationModel(config.model)
    optimizer = optimizer_factory(config.optimizer, model.parameters())
    model_l = LightingModelWrapperForMaskedLM(model=model, optimizer=optimizer)
    return config, data, model, optimizer, model_l


## PARSING WANDB SWEEP PARAMETERS


def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def parse_sweep_params(sweep_params):
    nested_params = {}
    for flat_key, value in sweep_params.items():
        keys = flat_key.split("__")
        temp_dict = value
        for key in reversed(keys):
            temp_dict = {key: temp_dict}
        nested_update(nested_params, temp_dict)
    return nested_params
