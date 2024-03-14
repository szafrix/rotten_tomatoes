from lightning.pytorch.callbacks import ModelCheckpoint

from classification.config.schemas import ModelCheckpointCallbackConfig


def model_checkpoint_callback(config: ModelCheckpointCallbackConfig) -> ModelCheckpoint:
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.save_top_k,
        monitor=config.monitor,
        mode=config.mode,
        auto_insert_metric_name=config.auto_insert_metric_name,
        save_weights_only=config.save_weights_only,
        every_n_train_steps=config.every_n_train_steps,
        dirpath=config.dirpath,
        filename=config.filename,
    )
    return checkpoint_callback
