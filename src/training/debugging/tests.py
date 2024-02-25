from copy import deepcopy
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from src.training.debugging.logger import get_logger

logger = get_logger()


class TrainingPipelineDebugger:

    def __init__(self, data, model_l):
        self.data = data
        self.model_l = model_l

    def test__fail_fast(self):
        data = deepcopy(self.data)
        model = deepcopy(self.model_l)
        trainer = Trainer(
            fast_dev_run=True,
            logger=False,
        )
        try:
            trainer.fit(model, datamodule=data)
        except Exception as exc:
            logger.error(f"FAIL: test__fail_fast, reason: {exc}")
        else:
            logger.info("PASS: test__fail_fast")

    def test_overfit_on_single_batch(self):
        data = deepcopy(self.data)
        model = deepcopy(self.model_l)
        data.config.batch_size = 2
        data.setup()
        trainer = Trainer(
            max_epochs=20,
            overfit_batches=1,
            log_every_n_steps=50,
            logger=False,
            # callbacks=[EarlyStopping(monitor="train/loss", stopping_threshold=0.0009)],
        )
        trainer.fit(model, datamodule=data)

        if train_loss := trainer.callback_metrics["train/loss"].item() <= 0.001:
            logger.info("PASS: test_overfit_on_single_batch")
        else:
            logger.error(
                f"FAIL: test_overfit_on_single_batch, final train loss: {train_loss}"
            )

    def test_overfit_on_single_batch(self):
        data = deepcopy(self.data)
        model = deepcopy(self.model_l)
        model.optimizer.param_groups[0]["lr"] = 2e-2
        data.config.batch_size = 2
        data.setup()
        trainer = Trainer(
            max_epochs=20,
            overfit_batches=1,
            log_every_n_steps=50,
            logger=False,
            # callbacks=[EarlyStopping(monitor="train/loss", stopping_threshold=0.0009)],
        )
        trainer.fit(model, datamodule=data)

        if train_loss := trainer.callback_metrics["train/loss"].item() <= 0.001:
            logger.info("PASS: test_overfit_on_single_batch")
        else:
            logger.error(
                f"FAIL: test_overfit_on_single_batch, final train loss: {train_loss}"
            )

    def batch_dependent_loss(self):
        raise NotImplementedError

    def test_dataset(self):
        raise NotImplementedError


def run_debugger(model_l, data):
    t = TrainingPipelineDebugger(data, model_l)
    t.test__fail_fast()
    t.test_overfit_on_single_batch()
