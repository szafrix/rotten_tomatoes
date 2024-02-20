import torch
from lightning.pytorch.callbacks.callback import Callback

from typing import List, Optional

from src.config.schemas import ModelConfig, OptimizerConfig, CallbacksConfig
from src.models.base_classes import BaseModel
from src.models.baselines import (
    InputIndependentBaselineModel,
    BaselineBERTLogisticRegressionModel,
)
from src.training.callbacks.model_checkpoint_callback import model_checkpoint_callback


def model_factory(config: ModelConfig) -> BaseModel:
    if config.name == "InputIndependentBaselineModel":
        return InputIndependentBaselineModel(config)
    if config.name == "BaselineBERTLogisticRegressionModel":
        return BaselineBERTLogisticRegressionModel(config)


def optimizer_factory(
    config: OptimizerConfig, model_parameters
) -> torch.optim.Optimizer:
    if lr := config.kwargs.get("lr"):
        config.kwargs["lr"] = float(lr)
    if betas := config.kwargs.get("betas"):
        config.kwargs["betas"] = eval(betas)
    return getattr(torch.optim, config.name)(params=model_parameters, **config.kwargs)


def callbacks_factory(config: CallbacksConfig) -> List[Optional[Callback]]:
    callbacks = []
    if hasattr(config, "model_checkpoint"):
        callbacks.append(model_checkpoint_callback(config.model_checkpoint))
    return callbacks
