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
import os
import time
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import auc, classification_report, roc_curve
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dataset import (BalancedSampler, ClassificationDataset, DatasetConfig,
                     Paper)
from experiment_io import log_to_file, make_output_folder

RESULT_FOLDER = "results"


@dataclass
class RelevanceModelConfig:
    """Configuration for the generator model."""

    model_name: str

    @classmethod
    def from_json(cls, json_file: str) -> "RelevanceModelConfig":
        """Load the configuration from a JSON file."""
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


@dataclass
class RelevanceModelTrainingConfig:
    learning_rate: float
    weight_decay: float
    num_workers: int
    batch_size: int
    num_epochs: int

    @classmethod
    def from_json(cls, json_file: str) -> "RelevanceModelTrainingConfig":
        """Load the configuration from a JSON file."""
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


class RelevanceModel:
    """
    Model wrapper class that handles training and evaluation of the relevance model.
    The relevance model predicts whether a paper is relevant or not given the title, abstract and authors.
    """

    def __init__(self, config: RelevanceModelConfig, experiment_name: str, verbose: bool):
        self.name = "RelevanceModel"
        self.output_folder = make_output_folder(
            os.path.join(experiment_name, "relevance_model"), verbose, remove=False
        )
        self.log_file = os.path.join(RESULT_FOLDER, experiment_name, "log.txt")
        self.config = config
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    def log(self, message: str) -> None:
        """
        Logs a message to the specified log file.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            tqdm.write(f"[ {self.name} ] {message}")
        log_to_file(self.log_file, self.name, message)

    def load_model(self) -> None:
        """
        Loads the relevance model and tokenizer from pre-trained weights.

        Args:
            None

        Returns:
            None
        """
        # NOTE(dave): Not supporting loading model from disk yet
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=2,
        )

    def train_epochs(
        self,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        num_epochs: int,
    ) -> None:
        """
        Trains the model for a specified number of epochs.

        Args:
            optimizer: The optimizer to use.
            criterion: The criterion to use.
            train_loader: The training data loader.
            model: The model to train.
            num_epochs: The number of epochs to train for.
        """
        model.train()
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        for epoch in tqdm(range(num_epochs), desc="Training Epochs", leave=False):
            total_loss = 0
            correct = 0
            total = 0

            for batch in tqdm(train_loader, desc="Batch", leave=False):
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                with torch.amp.autocast("cuda", enabled=True):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)
                    # loss = outputs.loss # not used because we scale the loss
                    logits = outputs.logits

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            self.log(
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss/len(train_loader)}, Accuracy: {correct/total}"
            )

    def finetune(
        self,
        dataset_config: DatasetConfig,
        training_config: RelevanceModelTrainingConfig,
    ) -> None:
        """
        Finetunes the model with the given dataset config and training config.

        Args:
            dataset_config: The dataset configuration.
            training_config: The training configuration.

        Returns:
            None
        """
        self.log(
            f"Training model {self.config.model_name} on dataset {dataset_config.classification_dataset_path}"
        )
        self.model.to(self.device)

        # optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=training_config.learning_rate,
            weight_decay=training_config.weight_decay,
        )

        # load dataset and split into train and eval
        train_dataset, eval_dataset = ClassificationDataset.load_train_eval_split(
            dataset_config, self.tokenizer
        )
        self.log(
            f"Dataset split: {len(train_dataset)} train samples, {len(eval_dataset)} eval samples"
        )

        # data loaders
        num_samples_minority = min(
            train_dataset.labels.count(0), train_dataset.labels.count(1)
        )  # perfectly balanced
        train_sampler = BalancedSampler(train_dataset.labels, num_samples_minority)
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config.batch_size,
            num_workers=training_config.num_workers,
            sampler=train_sampler,
        )

        # loss
        num_positives = sum([1 for data in train_loader.dataset.labels if data == 1])  # type: ignore
        num_negatives = len(train_loader.dataset) - num_positives  # type: ignore
        total_samples = num_positives + num_negatives
        class_weights_list = [total_samples / num_negatives, total_samples / num_positives]
        class_weights = torch.tensor(class_weights_list).to(self.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.log(f"Class weights: {class_weights}")

        self.log("Starting training")
        start_time = time.perf_counter()
        self.train_epochs(
            optimizer, criterion, train_loader, self.model, training_config.num_epochs
        )
        end_time = time.perf_counter()
        self.log(f"Finished training in {end_time - start_time:.2f} seconds, evaluating now")

        # save model
        self.log("Saving model")
        model_path = os.path.join(self.output_folder, "trained-relevancemodel-model.pt")
        torch.save(self.model, model_path)
        tokenizer_path = os.path.join(self.output_folder, "trained-relevancemodel-tokenizer.pt")
        torch.save(self.tokenizer, tokenizer_path)
        self.log(f"Model saved to {model_path}")
        self.log(f"Model saved to {tokenizer_path}")

    def evaluate(self, dataset_config: DatasetConfig) -> None:
        """
        Evaluate the relevance model on a given dataset.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset to evaluate on.

        Returns:
            None
        """
        self.log(
            f"Evaluate model {self.config.model_name} on dataset {dataset_config.classification_dataset_path}"
        )

        # load model from file
        model_path = os.path.join(self.output_folder, "trained-relevancemodel-model.pt")
        model = torch.load(model_path, weights_only=False)
        model.eval()

        tokenizer_path = os.path.join(self.output_folder, "trained-relevancemodel-tokenizer.pt")
        tokenizer = torch.load(tokenizer_path, weights_only=False)

        all_papers = Paper.load_list(
            dataset_config.classification_dataset_path, dataset_config.classification_data_type
        )
        dataset = ClassificationDataset(all_papers, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

        val_correct = 0
        val_total = 0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        all_scores: List[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="evaluation", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits

                preds = torch.argmax(logits, dim=1)
                scores = F.softmax(logits, dim=1)[:, 1]  # not sure if this is correct
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())

        self.log(f"Evaluation Accuracy: {val_correct/val_total}")
        print(classification_report(all_labels, all_preds))

        # plot ROC curve
        sns.set_style("darkgrid")
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.output_folder, "roc.png"))
