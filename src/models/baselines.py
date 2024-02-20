from torch import nn
from transformers import AutoModel
import numpy as np

from src.config.schemas import ModelConfig
from src.models.base_classes import BaseModel, PretrainedHuggingFaceModel


class InputIndependentBaselineModel(BaseModel):

    def __init__(self, config):
        super().__init__(config)

    def forward(self, x, labels=None):
        return np.random.choice([0, 1])


class BaselineBERTLogisticRegressionModel(PretrainedHuggingFaceModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cls_head = self.get_classification_head()

    def get_classification_head(self) -> nn.Sequential:
        return nn.Sequential(nn.Linear(768, 1))
