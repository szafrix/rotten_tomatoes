from dataclasses import dataclass

from typing import Optional, Dict, Any, Self

GLOBAL_SEED = 42


@dataclass
class DataConfigForMaskedLM:
    path_to_json: str
    path_to_save_dataset: str
    tokenizer_model_name: str
    tokenize_padding: bool
    tokenize_truncation: bool
    shuffle_upon_saving: bool
    drop_unnecessary_columns: bool
    single_chunk_size: int
    mlm_probability: float
    train_size: float
    batch_size: int
    num_workers_train: int
    num_workers_val: int
    shuffle_train: bool
    shuffle_val: bool


@dataclass
class ModelConfig:
    name: str
    hf_pretrained_model_name: Optional[str]


@dataclass
class OptimizerConfig:
    name: str
    kwargs: Optional[Dict[str, Any]]

@dataclass
class TrainingConfig:
    max_epochs: int
    accelerator: str
    check_val_every_n_epoch: int
    log_every_n_steps: int
    detect_anomaly: bool
