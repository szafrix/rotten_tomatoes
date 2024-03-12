from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets
from transformers import AutoTokenizer, DataCollatorWithPadding
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from src.config.schemas import DataConfig, GLOBAL_SEED


class RottenDataLoader(LightningDataModule):
    def __init__(self, config: DataConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_model_name)
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, return_tensors="pt"
        )

    def prepare_data(self):
        ds = load_dataset(self.config.hf_dataset_name)
        if self.config.path_to_scraped_json:
            ds["train"] = self.augment_train_dataset_with_scraped_quotes(
                ds["train"], self.config.path_to_scraped_json
            )
        ds = ds.map(self.tokenize, batched=True)
        if self.config.shuffle_upon_saving:
            ds = ds.shuffle(seed=GLOBAL_SEED)
        ds.save_to_disk(self.config.path_to_save_dataset)

    def setup(self, stage=None):
        dataset = DatasetDict.load_from_disk(self.config.path_to_save_dataset)
        self.dataset = dataset.remove_columns(["text"]).with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_train,
            shuffle=self.config.shuffle_train,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_val,
            shuffle=self.config.shuffle_val,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_test,
            shuffle=self.config.shuffle_test,
            collate_fn=self.data_collator,
        )

    def tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            padding=self.config.tokenize_padding,
            truncation=self.config.tokenize_truncation,
        )

    @staticmethod
    def augment_train_dataset_with_scraped_quotes(
        original_train_ds: Dataset, path_to_scraped_json: str
    ) -> Dataset:
        df_scraped = pd.read_json(path_to_scraped_json)
        ds_scraped = Dataset.from_pandas(df_scraped)
        ds_scraped = ds_scraped.remove_columns(["__index_level_0__"])
        ds_scraped = ds_scraped.cast(original_train_ds.features)
        return concatenate_datasets([original_train_ds, ds_scraped])
