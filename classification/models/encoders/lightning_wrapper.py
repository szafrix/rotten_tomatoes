import torch
from torch.optim import Optimizer

from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger

from classification.training.metrics.metrics import ModelMetrics
from classification.models.encoders.base_class import PretrainedHuggingFaceModel

from typing import Dict, Tuple, Optional


class LightingModelWrapperForBinaryClassification(LightningModule):

    def __init__(
        self, model: PretrainedHuggingFaceModel, optimizer: torch.optim.Optimizer
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits

        self.train_metrics = ModelMetrics(prefix="train/")
        self.val_metrics = ModelMetrics(prefix="val/")

        self.save_hyperparameters(ignore=["model"])
        # self.log_dict({"model_architecture": str(model.model)})

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(inputs)

    def on_train_start(self) -> None:
        self.train_metrics.to(self.device)
        self.val_metrics.to(self.device)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_metrics.compute())
        if isinstance(self.logger, WandbLogger):
            self.train_metrics.log_confusion_matrix(self.logger)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_metrics.compute())
        if isinstance(self.logger, WandbLogger):
            self.val_metrics.log_confusion_matrix(self.logger)
        self.val_metrics.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    # def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
    #    norms = grad_norm(self.model, norm_type=2)
    #    self.log_dict(norms)

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer

    def _step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, step_type: str
    ) -> torch.Tensor:
        inputs, targets = self._unpack_batch(batch)
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        preds = torch.sigmoid(outputs) > 0.5

        metrics = self._get_metrics_for_step_type(step_type)
        metrics.update(preds, targets, outputs)

        self.log(f"{step_type}/loss", loss, prog_bar=True, logger=True)

        return loss

    def _get_metrics_for_step_type(self, step_type: str) -> ModelMetrics:
        if step_type == "train":
            return self.train_metrics
        elif step_type == "val":
            return self.val_metrics
        else:
            raise NotImplementedError

    @staticmethod
    def _unpack_batch(
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
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