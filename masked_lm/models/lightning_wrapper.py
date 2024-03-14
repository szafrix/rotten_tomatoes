import torch
from torch.optim import Optimizer
from lightning import LightningModule

from masked_lm.models.architectures import RottenTomatoesDomainAdaptationModel
from typing import Dict


class LightingModelWrapperForMaskedLM(LightningModule):

    def __init__(
        self,
        model: RottenTomatoesDomainAdaptationModel,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model(inputs)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        raise NotImplementedError

    def predict_step(self, batch: Dict[str, torch.Tensor]) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> Optimizer:
        return self.optimizer

    def _step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int, step_type: str
    ) -> torch.Tensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log(f"{step_type}/loss", loss, prog_bar=True, logger=True)
        perplexity = torch.exp(loss)
        self.log(f"{step_type}/perplexity", perplexity, prog_bar=True, logger=True)
        return loss
