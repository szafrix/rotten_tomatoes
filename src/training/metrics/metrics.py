from torch import nn, Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryConfusionMatrix,
)

import wandb

import io
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns


class ModelMetrics(nn.Module):

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.prediction_metrics = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
            },
            prefix=prefix,
        )
        self.output_metrics = MetricCollection({"auroc": BinaryAUROC()}, prefix=prefix)
        self.confusion_matrix_metric = BinaryConfusionMatrix()

    def update(self, preds: Tensor, target: Tensor, raw_outputs: Tensor) -> None:
        self.prediction_metrics(preds, target)
        self.output_metrics(raw_outputs, target)
        self.confusion_matrix_metric(preds.int(), target.int())

    def compute(self):
        results = {**self.prediction_metrics.compute(), **self.output_metrics.compute()}
        return results

    def reset(self):
        self.prediction_metrics.reset()
        self.output_metrics.reset()
        self.confusion_matrix_metric.reset()

    def log_confusion_matrix(self, logger):
        cm = self.confusion_matrix_metric.compute().cpu().numpy()
        fig = self._plot_confusion_matrix(cm)
        img = self._fig_to_wandb_img(fig)
        logger.experiment.log({f"{self.prefix}confusion_matrix": [img]})

    def _plot_confusion_matrix(self, cm):
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{self.prefix}confusion matrix")
        ax.set_xlabel("Predicted labels")
        ax.set_ylabel("True labels")
        return fig

    @staticmethod
    def _fig_to_wandb_img(fig):
        with io.BytesIO() as buf:
            plt.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            wandb_img = wandb.Image(img)
        return wandb_img
