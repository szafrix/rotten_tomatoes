import torch
from transformers import BertForMaskedLM
from typing import Dict, Any
from masked_lm.config.schemas import ModelConfig


class RottenTomatoesDomainAdaptationModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model = BertForMaskedLM.from_pretrained(
            self.config.hf_pretrained_model_name
        )

    def __name__(self) -> None:
        return self.config.name

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        output = self.model(
            **{
                k: v
                for k, v in x.items()
                if k in ["input_ids", "attention_mask", "labels"]
            }
        )
        return output
