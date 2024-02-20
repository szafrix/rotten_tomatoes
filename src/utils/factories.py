import torch
from lightning.pytorch.callbacks.callback import Callback
import inspect
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
    optim_cls = getattr(torch.optim, config.name)
    valid_args = {
        k: v
        for k, v in config.kwargs.items()
        if k in inspect.getfullargspec(optim_cls.__init__).args
    }
    return optim_cls(params=model_parameters, **valid_args)


def callbacks_factory(config: CallbacksConfig) -> List[Optional[Callback]]:
    callbacks = []
    if hasattr(config, "model_checkpoint"):
        callbacks.append(model_checkpoint_callback(config.model_checkpoint))
    return callbacks