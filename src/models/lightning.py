from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
import io
import wandb
import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


class LightingModelWrapper(LightningModule):

    def __init__(self, model):
        assert issubclass(
            model.__class__, torch.nn.Module
        ), "Model is not subclass of nn.Module"
        super().__init__()
        self.model = model
        self.validation_step_outputs = []
        self.validation_step_targets = []
        self.loss_fn = torch.nn.functional.binary_cross_entropy_with_logits
        self.train_accuracy = BinaryAccuracy()
        self.train_auroc = BinaryAUROC()
        self.train_precision = BinaryPrecision()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()

        self.val_accuracy = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        # not logged metrics
        self.val_confusion_matrix = BinaryConfusionMatrix()

        self.save_hyperparameters()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, target = self._unpack_batch(batch)
        output = self(inputs)
        loss = self.loss_fn(output, target)
        predicted_labels = torch.where(
            output > 0.5, torch.tensor(1.0), torch.tensor(0.0)
        )
        self.train_accuracy(predicted_labels, target)
        self.train_auroc(output, target)
        self.train_precision(predicted_labels, target)
        self.train_recall(predicted_labels, target)
        self.train_f1(predicted_labels, target)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_epoch=True),
        self.log(
            "train/accuracy",
            self.train_accuracy,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "train/AUROC", self.train_auroc, prog_bar=False, logger=True, on_epoch=True
        )
        self.log(
            "train/precision",
            self.train_precision,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "train/recall",
            self.train_recall,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        self.log("train/f1", self.train_f1, prog_bar=False, logger=True, on_epoch=True)
        return loss

    # def on_train_epoch_end(self):
    #     all_preds = torch.stack(self.training_step_outputs)
    #     # do something with all preds
    #     ...
    #     self.training_step_outputs.clear()  # free memory

    def validation_step(self, batch, batch_idx):
        inputs, target = self._unpack_batch(batch)
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        predicted_labels = torch.where(
            output > 0.5, torch.tensor(1.0), torch.tensor(0.0)
        )
        # save/update attrs
        self.validation_step_outputs.append(output)
        self.validation_step_targets.append(target)
        self.log("val/loss", loss, prog_bar=True, logger=True),
        self.val_accuracy(predicted_labels, target)
        self.val_auroc(output, target)
        self.val_precision(predicted_labels, target)
        self.val_recall(predicted_labels, target)
        self.val_f1(predicted_labels, target)
        # log
        self.log("val/loss", loss, prog_bar=True, logger=True, on_epoch=True),
        self.log(
            "val/accuracy",
            self.val_accuracy,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "val/AUROC", self.val_auroc, prog_bar=False, logger=True, on_epoch=True
        )
        self.log(
            "val/precision",
            self.val_precision,
            prog_bar=False,
            logger=True,
            on_epoch=True,
        )
        self.log(
            "val/recall", self.val_recall, prog_bar=False, logger=True, on_epoch=True
        )
        self.log("val/f1", self.val_f1, prog_bar=False, logger=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs, dim=0)
        all_targets = torch.cat(self.validation_step_targets, dim=0)
        all_pred_labels = torch.where(
            all_preds > 0.5, torch.tensor(1.0), torch.tensor(0.0)
        )
        confusion_matrix = self.plot_confusion_matrix(all_pred_labels, all_targets)
        buf = io.BytesIO()
        confusion_matrix.savefig(buf, format="png")
        buf.seek(0)
        image = Image.open(buf)
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.log({"confusion_matrix": [wandb.Image(image)]})
        buf.close()
        plt.close(confusion_matrix)

        self.validation_step_outputs.clear()
        self.validation_step_targets.clear()

    def test_step(self, batch, batch_idx):
        inputs, target = self._unpack_batch(batch)
        output = self.model(inputs)
        loss = self.loss_fn(output, target)
        return loss

    def predict_step(self, batch):
        inputs, target = self._unpack_batch(batch)
        return self.model(inputs, batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

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

    def plot_confusion_matrix(self, preds, targets):
        mat = self.val_confusion_matrix(preds, targets).to("cpu")
        fig, ax = plt.subplots()
        sns.heatmap(mat, annot=True, cmap="Blues", fmt="g", ax=ax)
        return fig
