from lightning import LightningModule
import torch


class LightingModelWrapper(LightningModule):

    def __init__(self, model):
        assert issubclass(
            model.__class__, torch.nn.Module
        ), "Model is not subclass of nn.Module"
        super().__init__()
        self.model = model

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.BCELoss(output, target)

        # log metrics here
        self.log(
            "train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    # def on_train_epoch_end(self):
    #     all_preds = torch.stack(self.training_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.BCELoss(output, target)
        self.log("val_loss", loss)

    # def on_validation_epoch_end(self):
    #     all_preds = torch.stack(self.validation_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.validation_step_outputs.clear()  # free memory

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.BCELoss(output, target)
        # log metrics

    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)
