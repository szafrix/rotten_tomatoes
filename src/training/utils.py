import shutil
import argparse
from typing import Tuple, Any

from src.utils.factories import model_factory, optimizer_factory, callbacks_factory
from src.config.utils import parse_config
from src.data.data_loader import RottenDataLoader
from src.models.lightning_wrappers import (
    LightingModelWrapperForMulticlassClassification,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to .yaml file that contains configuration details",
    )
    return parser.parse_args()


def setup(config_path: str) -> Tuple[Any]:
    config = parse_config(config_path)
    data = RottenDataLoader(config.data)
    model = model_factory(config.model)
    optimizer = optimizer_factory(config.optimizer, model.parameters())
    callbacks = callbacks_factory(config.callbacks)
    model_l = LightingModelWrapperForMulticlassClassification(
        model=model, optimizer=optimizer
    )
    return config, data, model, optimizer, callbacks, model_l


def cleanup(model_ckpt_dir: str) -> None:
    shutil.rmtree(model_ckpt_dir)
