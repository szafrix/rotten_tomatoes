import shutil
import argparse
from typing import Tuple, Any

from src.utils.factories import model_factory, optimizer_factory, callbacks_factory
from src.config.utils import parse_config
from src.data.data_loader import RottenDataLoader
from src.models.classifiers.lightning_wrappers import (
    LightingModelWrapperForMulticlassClassification,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to .yaml file that contains configuration details",
    )
    parser.add_argument(
        "--run_debugger",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run some tests to check the training pipeline",
    )
    return parser.parse_args()


def setup(config_path: str, overwrite_dict=None) -> Tuple[Any]:
    config = parse_config(config_path, overwrite_dict)
    data = RottenDataLoader(config.data)
    model = model_factory(config.model)
    optimizer = optimizer_factory(config.optimizer, model.parameters())
    callbacks = callbacks_factory(config.callbacks)
    model_l = LightingModelWrapperForMulticlassClassification(
        model=model, optimizer=optimizer
    )
    return config, data, model, optimizer, callbacks, model_l


def cleanup(model_ckpt_dir: str) -> None:
    shutil.rmtree(model_ckpt_dir, ignore_errors=True)


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
