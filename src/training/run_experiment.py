from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger

from lightning import Trainer
import wandb
import torch
from dotenv import load_dotenv
from src.utils.wandb import wandb_login_ensure_personal_account
from src.training.utils import parse_args, setup, cleanup, parse_sweep_params
from src.training.debugging.tests import run_debugger

from dataclasses import asdict

seed_everything(42)
torch.set_float32_matmul_precision("high")

load_dotenv(override=True)


def main():
    args = parse_args()

    wandb_login_ensure_personal_account()
    wandb.init(project="rotten-tomatoes")
    if sweep_config := wandb.config:
        sweep_config = parse_sweep_params(sweep_config)

    config, data, model, optimizer, callbacks, model_l = setup(
        args.config_path, sweep_config
    )
    # data.prepare_data()
    data.setup()
    if args.run_debugger:
        run_debugger(model_l, data)

    wandb_logger = WandbLogger(
        project="rotten-tomatoes",
        tags=[model.__name__()],
        log_model=True,
    )
    wandb_logger.experiment.config["run_config"] = asdict(config)
    wandb_logger.watch(model, log="all")

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.training.accelerator,
        devices=1 if torch.cuda.is_available() else None,
        logger=wandb_logger,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch,
        log_every_n_steps=config.training.log_every_n_steps,
        callbacks=callbacks,
        detect_anomaly=config.training.detect_anomaly,
    )
    trainer.fit(model_l, datamodule=data)
    wandb.finish()

    cleanup(config.callbacks.model_checkpoint.dirpath)


if __name__ == "__main__":
    main()
