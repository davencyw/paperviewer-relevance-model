"""
This file is part of the Relevance Model System.
Copyright (C) 2025, David Schmidig, dave@davencyw.com

The Relevance Model System is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Relevance Model System is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with The Relevance Model System.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from db import PaperDatabaseManager
from stop_words import remove_stop_words


@dataclass
class Paper:
    """Datastructure to represent a paper"""

    title: str
    abstract: str
    authors: str
    label: Optional[bool]
    annotation: float = -2  # -2 means not annotated

    def get_preprocessed(self, disable_authors: bool = False) -> str:
        """
        Lowercase all string attributes of the paper and remove stop words
        Only used for classification!
        """
        self.title = self.title.lower()
        self.abstract = remove_stop_words(self.abstract)
        self.authors = self.authors.lower() if not disable_authors else ""
        title = self.title.replace("|", "")
        abstract = self.abstract.replace("|", "")
        return f"{title} | {abstract} | {self.authors}"

    @classmethod
    def load_list(cls, path: str, type: str) -> List["Paper"]:
        """
        Load a list of papers from a given path and type.

        Args:
            path: The path to the data source.
            type: The type of data source. Can be "json" or "db".

        Returns:
            A list of `Paper` objects.
        """
        if type == "json":
            return cls.list_from_json(path)
        elif type == "db":
            return cls.list_from_db(path)
        else:
            raise ValueError(f"Unknown type: {type}")
        return []  # comply with mypy

    @classmethod
    def list_from_json(cls, json_file: str) -> List["Paper"]:
        """
        Load the configuration from a JSON file.

        Args:
            json_file: The path to the JSON file containing the configuration.

        Returns:
            A list of `Paper` objects.
        """
        with open(json_file, "r") as file:
            data = json.load(file)
        return [cls(**paper) for paper in data]

    @classmethod
    def list_from_db(cls, db_file: str) -> List["Paper"]:
        """
        Load a list of papers from a given SQLite database file.

        Args:
            db_file: The path to the SQLite database file.

        Returns:
            A list of `Paper` objects.
        """
        db = PaperDatabaseManager(db_file)
        all_papers = db.get_all_papers()
        papers = []
        for paper in all_papers:
            # Create a new Paper instance from the database data
            # The label is the selected_review field in the database
            papers.append(
                cls(
                    title=paper.title,
                    abstract=paper.abstract,
                    authors=",".join(paper.authors),
                    label=paper.selected_review,
                )
            )
        return papers


@dataclass
class DatasetConfig:
    """
    Joint config over all datasets
    """

    # generator dataset
    generator_dataset_path: str
    generator_eval_percentage: float
    generator_data_type: str
    # classification dataset
    classification_dataset_path: str
    classification_eval_percentage: float
    classification_data_type: str
    max_samples_per_class: int

    @classmethod
    def from_json(cls, json_file: str) -> "DatasetConfig":
        """
        Load the configuration from a JSON file.

        Args:
            json_file (str): The path to the JSON file containing the configuration.

        Returns:
            DatasetConfig: The configuration loaded from the file.
        """
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


class GeneratorDataset(TorchDataset):
    def __init__(
        self, config: DatasetConfig, tokenizer, max_length: int = 512, verbose: bool = True
    ):
        """
        Initialize the dataset for the generator model.

        Args:
            config (DatasetConfig): The configuration for the dataset.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_length (int, optional): The maximum length of the input text in tokens. Defaults to 512.
            verbose (bool, optional): Whether to print information about the dataset. Defaults to True.
        """
        self.papers = Paper.load_list(config.generator_dataset_path, config.generator_data_type)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.verbose = verbose

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        paper = self.papers[idx]
        # construct the actual prompt completion
        text = f"The next Paper is | {paper.title} | {paper.abstract} |"
        tokenized = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
        }


class RewardDataset(TorchDataset):
    def __init__(self, papers: List[Paper], tokenizer, max_length: int = 512):
        """
        Initialize the dataset for the reward model.

        Args:
            papers (List[Paper]): The list of papers to use.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_length (int, optional): The maximum length of the input text in tokens. Defaults to 512.
        """
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        paper = self.papers[idx]
        text = f"{paper.title} | {paper.abstract}"
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "annotation": torch.tensor(paper.annotation, dtype=torch.float),
        }


class ClassificationDataset(TorchDataset):
    @classmethod
    def load_train_eval_split(cls, config: DatasetConfig, tokenizer):
        """
        Load the dataset from a JSON file and split it into a training and evaluation set.

        Args:
            config (DatasetConfig): The configuration for the dataset.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

        Returns:
            (ClassificationDataset, ClassificationDataset): The training and evaluation datasets.
        """
        all_papers = Paper.load_list(
            config.classification_dataset_path, config.classification_data_type
        )
        random.shuffle(all_papers)
        train_size = int(len(all_papers) * (1.0 - config.classification_eval_percentage))
        train_dataset = ClassificationDataset(all_papers[:train_size], tokenizer)
        test_dataset = ClassificationDataset(all_papers[train_size:], tokenizer)
        return train_dataset, test_dataset

    def __init__(self, papers: List[Paper], tokenizer, max_length: int = 512, verbose: bool = True):
        """
        Initialize the dataset for the classification model.

        Args:
            papers (List[Paper]): The list of papers to use.
            tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
            max_length (int, optional): The maximum length of the input text in tokens. Defaults to 512.
            verbose (bool, optional): Whether to print information about the dataset. Defaults to True.
        """
        self.papers = papers
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.verbose = verbose
        self.labels = [paper.label for paper in self.papers]

    def __len__(self):
        return len(self.papers)

    def __getitem__(self, idx):
        # randomly disable authors 50% of time
        # TODO(dave): refactor to param
        disable_authors = random.random() < 0.5
        text = self.papers[idx].get_preprocessed(disable_authors)
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": text,
        }


class BalancedSampler(torch.utils.data.Sampler):
    def __init__(self, labels, num_negative_samples_per_epoch=float("inf")):
        """
        Sampler that samples a balanced proportion of positive and negative samples.

        Args:
            labels (List[int]): The labels of the dataset.
            num_negative_samples_per_epoch (int, optional): The number of negative samples to sample per epoch. Defaults to infinity.
        """
        self.name = "BalancedSampler"
        self.labels = labels
        self.num_negative_samples_per_epoch = num_negative_samples_per_epoch

        # Indices of positive and negative samples
        self.positive_indices = [i for i, label in enumerate(labels) if label == 1]
        self.negative_indices = [i for i, label in enumerate(labels) if label == 0]
        self.log(
            f"Initialized sampler with {num_negative_samples_per_epoch} negative samples per epoch"
        )

    def log(self, message: str):
        tqdm.write(f"[ {self.name} ] {message}")

    def __iter__(self):
        # Sample all positive samples
        positive_samples = self.positive_indices

        # Randomly sample the specified number of negative samples
        # check if num_negative_samples_per_epochs is not infinite
        if self.num_negative_samples_per_epoch == float("inf"):
            negative_samples = self.negative_indices
        else:
            negative_samples = torch.randperm(len(self.negative_indices))[
                : self.num_negative_samples_per_epoch
            ]
            negative_samples = [self.negative_indices[i] for i in negative_samples]

        # Combine positive and negative samples
        combined_samples = positive_samples + negative_samples

        # Shuffle the combined samples
        combined_samples = torch.randperm(len(combined_samples)).tolist()

        return iter(combined_samples)

    def __len__(self):
        # Return the total number of sampled data
        total_samples = len(self.positive_indices)
        if self.num_negative_samples_per_epoch == float("inf"):
            total_samples += len(self.negative_indices)
        else:
            total_samples += +self.num_negative_samples_per_epoch
        return total_samples
