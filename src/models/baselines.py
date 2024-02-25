from torch import nn
import torch.nn.init as init
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
        init.xavier_uniform_(self.cls_head[-1].weight)
        init.zeros_(self.cls_head[-1].bias)

    def get_classification_head(self) -> nn.Sequential:
        return nn.Sequential(nn.Dropout(0.5), nn.Linear(768, 1))
