import torch
from transformers import AutoModelForMaskedLM, AdamW

from masked_lm.config.schemas import ModelConfig


class RottenTomatoesDomainAdaptationModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.config.hf_pretrained_model_name
        )

    def __name__(self):
        return self.config.name

    def forward(self, x):
        output = self.model(
            **{
                k: v
                for k, v in x.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }
        )
        return output
