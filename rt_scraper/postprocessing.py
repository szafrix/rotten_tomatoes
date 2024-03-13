import regex as re

from datasets import load_dataset
import pandas as pd
import numpy as np
from typing import List, Iterable
from rapidfuzz import fuzz
from tqdm import tqdm


def load_original_rt_train_dataset():
    ds = load_dataset("rotten_tomatoes")
    train = ds["train"].to_pandas()
    val = ds["validation"].to_pandas()
    test = ds["test"].to_pandas()
    return train, val, test


def add_whitespaces_before_and_after_punctuation_marks(text):
    pattern = r"([\p{P}])"
    text_with_whitespaces = re.sub(pattern, r" \1 ", text)
    multiple_whitespaces_pattern = r"\s{2,}"
    text_with_single_whitespaces = re.sub(
        multiple_whitespaces_pattern, " ", text_with_whitespaces
    )
    return text_with_single_whitespaces.strip()


def is_sample_present_in_original_dataset(
    sample: str, original_dataset_samples: List[str]
) -> str | None:
    threshold = 80
    for sample_from_original in original_dataset_samples:
        if fuzz.QRatio(sample, sample_from_original) > threshold:
            print(sample_from_original, sample)
            return True
    return False


def find_samples_already_present_in_original_dataset(
    new_samples: Iterable[str], all_texts_from_original_dataset: Iterable[str]
):
    spotted = []
    to_drop = []
    for new_sample in tqdm(new_samples):
        for orig_text in all_texts_from_original_dataset:
            if is_sample_present_in_original_dataset(new_sample, orig_text):
                spotted.append((new_sample, orig_text))
                to_drop.append(new_sample)
    return spotted, to_drop


def preprocess_scraped_data_to_match_original_dataset(df):
    df = df[["quote", "score"]].copy()
    df.columns = ["text", "label"]
    df["label"] = df["label"].replace({"fresh": 1, "rotten": 0}).copy()

    # lower
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].apply(add_whitespaces_before_and_after_punctuation_marks)
    return df

    # whitespace before special signs


if __name__ == "__main__":
    train, val, test = load_original_rt_train_dataset()
    new_train_dataset = pd.read_json("rt_scraper/results/scraped_dataset.json")
    new_train_dataset = preprocess_scraped_data_to_match_original_dataset(
        new_train_dataset
    )
    all_texts_from_original_dataset = (
        train["text"].tolist() + val["text"].tolist() + test["text"].tolist()
    )
    spotted, to_drop = find_samples_already_present_in_original_dataset(
        new_train_dataset["text"].values, all_texts_from_original_dataset
    )
    new_train_dataset[~new_train_dataset["text"].isin(to_drop)].to_json(
        "rt_scraper/results/scraped_dataset_sans_already_present_quotes.json"
    )
