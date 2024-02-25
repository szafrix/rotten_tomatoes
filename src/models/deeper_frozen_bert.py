from torch import nn
import torch.nn.init as init
from src.models.base_classes import PretrainedHuggingFaceModel
from src.config.schemas import ModelConfig


class FrozenBERTWithDeeperHead(PretrainedHuggingFaceModel):

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cls_head = self.get_classification_head()

    def get_classification_head(self) -> nn.Sequential:
        z1 = nn.Linear(768, 1024)
        init.kaiming_uniform_(z1.weight)
        init.normal_(z1.bias)
        bn1 = nn.BatchNorm1d(1024)
        a1 = nn.ReLU()
        # d1 = nn.Dropout(0.2)
        z2 = nn.Linear(1024, 512)
        init.kaiming_uniform_(z2.weight)
        init.normal_(z2.bias)
        bn2 = nn.BatchNorm1d(512)
        a2 = nn.ReLU()
        # d2 = nn.Dropout(0.2)
        z3 = nn.Linear(512, 32)
        init.kaiming_uniform_(z3.weight)
        init.normal_(z3.bias)
        bn3 = nn.BatchNorm1d(32)
        a3 = nn.ReLU()
        # d3 = nn.Dropout(0.2)
        z4 = nn.Linear(32, 1)
        init.xavier_uniform_(z4.weight)
        init.normal_(z4.bias)
        return nn.Sequential(z1, bn1, a1, z2, bn2, a2, z3, bn3, a3, z4)
