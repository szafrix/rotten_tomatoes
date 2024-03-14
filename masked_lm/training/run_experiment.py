import os
import sys
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.append(os.environ.get("PROJECT_DIR"))

from lightning import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import callbacks

import torch
from lightning import Trainer
import wandb

from masked_lm.utils.wandb import wandb_login_ensure_personal_account
from masked_lm.training.utils import parse_args, setup, parse_sweep_params

from dataclasses import asdict

seed_everything(42)
torch.set_float32_matmul_precision("high")


def main():
    args = parse_args()

    wandb_login_ensure_personal_account()
    wandb.init(project="rotten-tomatoes")
    if sweep_config := wandb.config:
        sweep_config = parse_sweep_params(sweep_config)

    config, data, model, optimizer, model_l = setup(args.config_path, sweep_config)
    # data.prepare_data()
    data.setup()

    wandb_logger = WandbLogger(
        project="rotten-tomatoes",
        tags=[config.model.name],
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
        detect_anomaly=config.training.detect_anomaly,
        callbacks=callbacks.ModelCheckpoint(save_top_k=1),
    )
    trainer.fit(model_l, datamodule=data)
    wandb.finish()

    model_l.model.model.save_pretrained("./masked_lm/models/best_model", from_pt=True)


if __name__ == "__main__":
    main()
