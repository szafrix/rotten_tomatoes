import torch
from classification.config.schemas import ModelConfig
from transformers import AutoModel


class BaseModel(torch.nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    def __name__(self):
        return self.config.name


class PretrainedHuggingFaceModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self.load_model_from_hf()
        self.cls_head = self.get_classification_head()

    def forward(self, x):
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

    def load_model_from_hf(self):
        model = AutoModel.from_pretrained(self.config.hf_pretrained_model_name)
        if self.config.freeze_weights:
            for p in model.parameters():
                p.requires_grad = False
                p.grad = None
        return model

    def get_classification_head(self) -> torch.nn.Sequential:
        pass


class PretrainedHuggingFaceModelAfterDomainAdaptation(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = self.load_model_from_file()
        self.cls_head = self.get_classification_head()

    def forward(self, x):
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

    def load_model_from_file(self):
        model = AutoModel.from_pretrained(self.config.hf_pretrained_model_name)
        model_path = "masked_lm/models/best_model"
        model = model.from_pretrained(model_path)
        if self.config.freeze_weights:
            for p in model.parameters():
                p.requires_grad = False
                p.grad = None
        return model

    def get_classification_head(self) -> torch.nn.Sequential:
        pass
