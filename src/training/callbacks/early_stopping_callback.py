from lightning.pytorch.callbacks import EarlyStopping
from src.config.schemas import EarlyStoppingCallbackConfig


def early_stopping_callback(config: EarlyStoppingCallbackConfig):
    return EarlyStopping(
        monitor=config.monitor, patience=config.patience, mode=config.mode
    )
