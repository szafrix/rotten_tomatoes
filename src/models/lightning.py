import torch
from lightning import LightningModule, seed_everything
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.utilities import grad_norm
from torch.optim.optimizer import Optimizer

from src.models.metrics import ModelMetrics

seed_everything(42)


class LightingModelWrapper(LightningModule):

    def __init__(self, model):
        assert issubclass(
            model.__class__, torch.nn.Module
        ), "Model is not subclass of nn.Module"
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        self.train_metrics = ModelMetrics(
            prefix="train/"
        )  # TODO: move to on train start
        self.val_metrics = ModelMetrics(prefix="val/")

        self.save_hyperparameters(ignore=["model"])

    def on_train_start(self):
        self.train_metrics.to(self.device)
        self.val_metrics.to(self.device)

    def forward(self, inputs):
        return self.model(inputs)

    def _step(self, batch, batch_idx, step_type):
        inputs, targets = self._unpack_batch(batch)
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = torch.sigmoid(outputs) > 0.5

        metrics = self._get_proper_metrics(step_type)
        metrics.update(preds, targets, outputs)

        self.log(f"{step_type}/loss", loss, prog_bar=True, logger=True)

        return loss

    def _get_proper_metrics(self, step_type):
        if step_type == "train":
            return self.train_metrics
        elif step_type == "val":
            return self.val_metrics
        else:
            raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        if isinstance(self.logger, WandbLogger):
            self.train_metrics.log_confusion_matrix(self.logger)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        if isinstance(self.logger, WandbLogger):
            self.val_metrics.log_confusion_matrix(self.logger)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch):
        raise NotImplementedError

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #    norms = grad_norm(self.model, norm_type=2)
    #    self.log_dict(norms)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-2)

    @staticmethod
    def _unpack_batch(batch):
        inputs = {
            k: v
            for k, v in batch.items()
            if k in ["input_ids", "token_type_ids", "attention_mask"]
        }
        if "labels" in batch.keys():
            targets = batch["labels"].view(-1, 1).float()
        else:
            targets = None
        return inputs, targets
