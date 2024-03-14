import torch
from lightning.pytorch.callbacks.callback import Callback
import inspect
from typing import List, Optional

from classification.config.schemas import ModelConfig, OptimizerConfig, CallbacksConfig
from classification.models.encoders.base_class import PretrainedHuggingFaceModel
from classification.models.encoders.architectures import (
    BERTWithSimpleHead,
    BERTWithDeeperHead,
)
from classification.training.callbacks.model_checkpoint_callback import (
    model_checkpoint_callback,
)
from classification.training.callbacks.early_stopping_callback import (
    early_stopping_callback,
)


def model_factory(config: ModelConfig) -> PretrainedHuggingFaceModel:
    if "SimpleHead" in config.name:
        return BERTWithSimpleHead(config)
    if "DeeperHead" in config.name:
        return BERTWithDeeperHead(config)


def optimizer_factory(
    config: OptimizerConfig, model_parameters
) -> torch.optim.Optimizer:
    if lr := config.kwargs.get("lr"):
        config.kwargs["lr"] = float(lr)
    if (beta1 := config.kwargs.get("beta1")) and (beta2 := config.kwargs.get("beta2")):
        config.kwargs["betas"] = (float(beta1), float(beta2))
    optim_cls = getattr(torch.optim, config.name)
    valid_args = {
        k: v
        for k, v in config.kwargs.items()
        if k in inspect.getfullargspec(optim_cls.__init__).args
    }
    return optim_cls(params=model_parameters, **valid_args)


def callbacks_factory(config: CallbacksConfig) -> List[Optional[Callback]]:
    callbacks = []
    if config.model_checkpoint:
        callbacks.append(model_checkpoint_callback(config.model_checkpoint))
    if config.early_stopping:
        callbacks.append(early_stopping_callback(config.early_stopping))
    return callbacks
