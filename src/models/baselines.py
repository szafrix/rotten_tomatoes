from torch import nn
from transformers import AutoModel
import numpy as np


class InputIndependentBaselineModel(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, labels=None):
        return np.random.choice([0, 1])


class BaselineBERTLogisticRegressionModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(self.config["baseline_model_checkpoint"])
        for p in self.model.parameters():
            p.requires_grad = False
        self.logreg = nn.Linear(768, 1)

    def forward(self, x):
        bert_output = self.model(
            **{
                k: v
                for k, v in x.items()
                if k in ["input_ids", "token_type_ids", "attention_mask"]
            }
        )
        x = bert_output.last_hidden_state[:, 0, :]
        x = self.logreg(x)
        return x

    def __name__(self):
        return "BaselineBERTLogisticRegressionModel"
