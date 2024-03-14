import torch
from lightning import LightningModule

from masked_lm.models.architectures import RottenTomatoesDomainAdaptationModel


class LightingModelWrapperForMaskedLM(LightningModule):

    def __init__(
        self,
        model: RottenTomatoesDomainAdaptationModel,
        optimizer: torch.optim.Optimizer,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.save_hyperparameters(ignore=["model"])

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

    def predict_step(self, batch):
        raise NotImplementedError

    def configure_optimizers(self):
        return self.optimizer

    def _step(self, batch, batch_idx, step_type):
        outputs = self(batch)
        loss = outputs.loss
        self.log(f"{step_type}/loss", loss, prog_bar=True, logger=True)
        perplexity = torch.exp(loss)
        self.log(f"{step_type}/perplexity", perplexity, prog_bar=True, logger=True)
        return loss
