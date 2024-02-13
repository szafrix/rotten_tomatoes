from lightning.pytorch import callbacks


def model_checkpoint_callback():
    checkpoint_callback = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="val/loss",
        mode="min",
        auto_insert_metric_name=False,
        save_weights_only=False,
        every_n_train_steps=None,
        dirpath="model_ckpts/",
        filename="{epoch}-{val/loss:.2f}",
    )
    return checkpoint_callback
