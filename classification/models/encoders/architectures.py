from torch import nn
import torch.nn.init as init
from transformers import AutoModel
import numpy as np

from classification.config.schemas import ModelConfig
from classification.models.encoders.base_class import PretrainedHuggingFaceModel


class BERTWithSimpleHead(PretrainedHuggingFaceModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cls_head = self.get_classification_head()
        init.xavier_uniform_(self.cls_head[-1].weight)
        init.zeros_(self.cls_head[-1].bias)

    def get_classification_head(self) -> nn.Sequential:
        return nn.Sequential(nn.Linear(768, 1))


class BERTWithDeeperHead(PretrainedHuggingFaceModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cls_head = self.get_classification_head()

    def get_classification_head(self) -> nn.Sequential:
        z1 = nn.Linear(768, 1024)
        init.kaiming_uniform_(z1.weight)
        init.normal_(z1.bias)
        bn1 = nn.BatchNorm1d(1024)
        a1 = nn.ReLU()
        z2 = nn.Linear(1024, 512)
        init.kaiming_uniform_(z2.weight)
        init.normal_(z2.bias)
        bn2 = nn.BatchNorm1d(512)
        a2 = nn.ReLU()
        z3 = nn.Linear(512, 32)
        init.kaiming_uniform_(z3.weight)
        init.normal_(z3.bias)
        bn3 = nn.BatchNorm1d(32)
        a3 = nn.ReLU()
        z4 = nn.Linear(32, 1)
        init.xavier_uniform_(z4.weight)
        init.normal_(z4.bias)
        return nn.Sequential(z1, bn1, a1, z2, bn2, a2, z3, bn3, a3, z4)
