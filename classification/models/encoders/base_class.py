import torch
from classification.config.schemas import ModelConfig
from transformers import AutoModel

from typing import Dict, Any


class PretrainedHuggingFaceModel(torch.nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if self.config.domain_adapted_model_path:
            self.model = self.load_model_from_file()
        else:
            self.model = self.load_model_from_hf()
        self.cls_head = self.get_classification_head()

    def forward(self, x: Dict[str, Any]) -> torch.Tensor:
        output = self.model(
            **{
                k: v
                for k, v in x.items()
                if k in ["input_ids", "token_type_ids", "attention_mask"]
            }
        )
        x = output.last_hidden_state[:, 0, :]
        x = self.cls_head(x)
        return x

    def load_model_from_hf(self) -> AutoModel:
        model = AutoModel.from_pretrained(self.config.hf_pretrained_model_name)
        if self.config.freeze_weights:
            for p in model.parameters():
                p.requires_grad = False
                p.grad = None
        return model

    def load_model_from_file(self) -> AutoModel:
        model = AutoModel.from_pretrained(self.config.hf_pretrained_model_name)
        model = model.from_pretrained(self.config.domain_adapted_model_path)
        if self.config.freeze_weights:
            for p in model.parameters():
                p.requires_grad = False
                p.grad = None
        return model

    def get_classification_head(self) -> torch.nn.Sequential:
        pass
