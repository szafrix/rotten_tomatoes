import lightning as L
from torch import nn
from transformers import AutoModel


class InputIndependentBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 1


class BaselineBERTLogisticRegressionModel(nn.Module):

    def __init__(self, config):
        self.config = config
        self.model = AutoModel.from_pretrained(self.config["baseline_model_checkpoint"])

    def forward():
        pass
