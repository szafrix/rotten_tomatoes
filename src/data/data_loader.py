from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from lightning import LightningDataModule
from torch.utils.data import DataLoader
import torch

torch.manual_seed(42)
torch.cuda.manual_seed(42)


class RottenDataLoader(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_checkpoint"])
        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, return_tensors="pt"
        )

    def prepare_data(self):
        ds = load_dataset(self.config["dataset_name"])
        ds = ds.map(self.tokenize, batched=True)
        ds.save_to_disk(self.config["dataset_local_path"])

    def setup(self, stage=None):
        dataset = DatasetDict.load_from_disk(self.config["dataset_local_path"])
        self.dataset = dataset.remove_columns(["text"]).with_format("torch")

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.config["batch_size"],
            num_workers=31,  # TODO play with it
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.config["batch_size"],
            num_workers=31,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.config["batch_size"],
            num_workers=31,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def tokenize(self, batch):
        return self.tokenizer(batch["text"], padding=False, truncation=True)
