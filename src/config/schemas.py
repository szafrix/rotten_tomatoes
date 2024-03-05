from dataclasses import dataclass

from typing import Optional, Dict, Any, Self

GLOBAL_SEED = 42


@dataclass
class DataConfig:
    hf_dataset_name: str
    path_to_save_dataset: str
    tokenizer_model_name: str
    shuffle_upon_saving: bool
    tokenize_padding: bool
    tokenize_truncation: bool
    batch_size: int
    num_workers_train: int
    num_workers_val: int
    num_workers_test: int
    shuffle_train: bool
    shuffle_val: bool
    shuffle_test: bool


@dataclass
class ModelConfig:
    name: str
    hf_pretrained_model_name: Optional[str]  # for HF based models
    freeze_weights: bool  # for HF based models


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


@dataclass
class ModelCheckpointCallbackConfig:
    save_top_k: int
    monitor: str
    mode: str
    auto_insert_metric_name: bool
    save_weights_only: bool
    every_n_train_steps: Optional[int]
    dirpath: str
    filename: str


@dataclass
class EarlyStoppingCallbackConfig:
    monitor: str
    patience: int
    mode: str


@dataclass
class CallbacksConfig:
    model_checkpoint: ModelCheckpointCallbackConfig
    early_stopping: Optional[EarlyStoppingCallbackConfig]

    @classmethod
    def from_dict(cls, obj: Dict) -> Self:
        if obj.get("early_stopping"):
            return cls(
                model_checkpoint=ModelCheckpointCallbackConfig(
                    **obj["model_checkpoint"]
                ),
                early_stopping=EarlyStoppingCallbackConfig(**obj["early_stopping"]),
            )
        else:
            return cls(
                model_checkpoint=ModelCheckpointCallbackConfig(
                    **obj["model_checkpoint"]
                ),
            )


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    callbacks: CallbacksConfig

    @classmethod
    def from_dict(cls, obj: Dict) -> Self:
        return cls(
            data=DataConfig(**obj["data"]),
            model=ModelConfig(**obj["model"]),
            optimizer=OptimizerConfig(**obj["optimizer"]),
            training=TrainingConfig(**obj["training"]),
            callbacks=CallbacksConfig.from_dict(obj["callbacks"]),
        )
