from lightning.pytorch.callbacks import EarlyStopping
from classification.config.schemas import EarlyStoppingCallbackConfig


def early_stopping_callback(config: EarlyStoppingCallbackConfig) -> EarlyStopping:
    return EarlyStopping(
        monitor=config.monitor, patience=config.patience, mode=config.mode
    )
