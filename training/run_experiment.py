from lightning.pytorch import callbacks
from pathlib import Path


def main():
    monitor_metric = "validation/loss"
    
    Path("training/logs").mkdir(parents=True, exist_ok=True)
    checkpoint_callback = callbacks.ModelCheckpoint(save_top_k=3,
                                                    filename="epoch={epoch:04d}-validation.loss={validation/loss:.3f}",
                                                    monitor=monitor_metric,
                                                    mode="min",
                                                    auto_insert_metric_name=False,
                                                    save_weights_only=False,
                                                    every_n_train_steps= None,
                                                    dirpath="training/logs"
                                                    )
    
    summary_callback = callbacks.ModelSummary(max_depth=2)
    callbacks = [checkpoint_callback, summary_callback]
    