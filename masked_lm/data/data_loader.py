from datasets import DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import pandas as pd
from masked_lm.config.schemas import DataConfigForMaskedLM, GLOBAL_SEED


class RottenDataLoaderForMaskedLM(LightningDataModule):
    def __init__(self, config: DataConfigForMaskedLM):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_model_name, use_fast=True
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.config.mlm_probability,
            return_tensors="pt",
        )

    def prepare_data(self):
        df_scraped = pd.read_json(self.config.path_to_json)
        ds_scraped = Dataset.from_pandas(df_scraped)
        ds_scraped = ds_scraped.remove_columns(["__index_level_0__"])
        ds = ds_scraped.map(self.tokenize, batched=True)
        if self.config.shuffle_upon_saving:
            ds = ds.shuffle(seed=GLOBAL_SEED)
        if self.config.drop_unnecessary_columns:
            ds = ds.remove_columns(["text", "label", "token_type_ids"])
        ds = self.split_dataset_into_equally_sized_chunks(ds)
        ds = ds.train_test_split(train_size=self.config.train_size, seed=GLOBAL_SEED)
        ds.save_to_disk(self.config.path_to_save_dataset)

    def setup(self, stage=None):
        dataset = DatasetDict.load_from_disk(self.config.path_to_save_dataset)
        dataset = dataset.with_format("torch")
        return dataset

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
            self.dataset["test"],
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers_val,
            shuffle=self.config.shuffle_val,
            collate_fn=self.data_collator,
        )

    def tokenize(self, batch):
        batch_tokenized = self.tokenizer(
            batch["text"],
            padding=self.config.tokenize_padding,
            truncation=self.config.tokenize_truncation,
        )
        if self.tokenizer.is_fast:
            batch_tokenized["word_ids"] = [
                batch_tokenized.word_ids(i)
                for i in range(len(batch_tokenized["input_ids"]))
            ]
        return batch_tokenized

    def split_dataset_into_equally_sized_chunks(self, batch):
        concatenated_batch = {k: sum(batch[k], []) for k in batch.keys()}
        chunks = self._split_concatenated_batch_into_equal_chunks(concatenated_batch)
        last_chunk_size = len(chunks["input_ids"][-1])
        if last_chunk_size != self.config.single_chunk_size:
            chunks = self._drop_last_chunk(chunks)
        chunks["labels"] = chunks["input_ids"].copy()
        return chunks

    def _split_concatenated_batch_into_equal_chunks(self, concatenated_batch):
        total_length = len(concatenated_batch["input_ids"])
        chunks = {
            column: [
                values[i : i + self.config.single_chunk_size]
                for i in range(0, total_length, self.config.single_chunk_size)
            ]
            for column, values in concatenated_batch.items()
        }
        return chunks

    def _drop_last_chunk(self, chunks):
        return {column: values[:-1] for column, values in chunks.items()}
