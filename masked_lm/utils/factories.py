import torch
import inspect
from masked_lm.config.schemas import OptimizerConfig


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
