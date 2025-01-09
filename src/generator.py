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
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import random_split
from tqdm import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorForLanguageModeling, GPT2Config,
                          GPT2LMHeadModel, GPT2Model, GPT2PreTrainedModel,
                          GPT2Tokenizer, Trainer, TrainingArguments, pipeline)
from trl import PPOConfig, PPOTrainer, create_reference_model

from dataset import (ClassificationDataset, DatasetConfig, GeneratorDataset,
                     Paper, RewardDataset)
from experiment_io import log_to_file, make_output_folder
from review_ui import TextAnnotationTUI

RESULT_FOLDER = "results"


@dataclass
class GeneratorConfig:
    """Configuration for the generator model."""

    model_name: str

    @classmethod
    def from_json(cls, json_file: str) -> "GeneratorConfig":
        """Load the configuration from a JSON file."""
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


@dataclass
class GeneratorTrainingConfig:
    learning_rate: float
    output_folder: str
    per_device_batch_size: int
    num_train_epochs: int
    weight_decay: float
    warmup_step: int
    logging_steps: int
    fp16: bool
    push_to_hub: bool

    @classmethod
    def from_json(cls, json_file: str) -> "GeneratorTrainingConfig":
        """Load the configuration from a JSON file."""
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


@dataclass
class RLHFTrainingConfig:
    reward_num_epochs: int
    reward_per_device_train_batch_size: int
    reward_save_steps: int
    reward_save_total_limit: int
    reward_logging_steps: int
    reward_learning_rate: float
    reward_weight_decay: float
    ppo_num_epochs: int
    ppo_per_device_train_batch_size: int
    ppo_save_steps: int
    ppo_save_total_limit: int
    ppo_logging_steps: int
    ppo_learning_rate: float
    ppo_weight_decay: float
    value_model_num_epochs: int
    value_model_per_device_train_batch_size: int
    value_model_save_steps: int
    value_model_save_total_limit: int
    value_model_logging_steps: int
    value_model_learning_rate: float
    value_model_weight_decay: float
    sampling_top_k: int
    sampling_temperature: float
    sampling_top_p: float

    @classmethod
    def from_json(cls, json_file: str) -> "RLHFTrainingConfig":
        """Load the configuration from a JSON file."""
        with open(json_file, "r") as file:
            data = json.load(file)
        return cls(**data)


class RewardModel(GPT2PreTrainedModel):
    """
    The reward model that is used to run PPO.
    """

    def __init__(self, model_name="gpt2"):
        super().__init__(GPT2Config.from_pretrained("gpt2"))
        self.transformer = GPT2Model.from_pretrained(model_name)
        self.head = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        annotation: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs the reward model and computes the loss.

        Args:
            input_ids: The input IDs to score.
            attention_mask: The attention mask for the input IDs. If None, use all ones.
            annotation: The annotation scores for the input IDs.

        Returns:
            A dictionary containing the loss and logits.
        """
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        rewards = self.head(cls_token_state).squeeze(-1)
        loss_fn = nn.MSELoss()
        loss = loss_fn(rewards, annotation)
        return {"loss": loss, "logits": rewards}

    def score(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Score a tokenized input with the reward model.

        Args:
            input_ids: The input IDs to score.
            attention_mask: The attention mask for the input IDs. If None, use all ones.

        Returns:
            A tensor containing the reward scores for each input ID.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_token_state = outputs.last_hidden_state[:, 0, :]
        rewards = self.head(cls_token_state).squeeze(-1)
        return rewards

    @classmethod
    def load_model_from_disk(cls, file_path: str, model_name: str) -> "RewardModel":
        """
        Load a model from disk.

        Args:
            file_path: The path to the saved model.
            model_name: The name of the model to load.

        Returns:
            The loaded model.
        """
        model = cls(model_name=model_name)
        model.load_state_dict(torch.load(file_path))
        return model

    @classmethod
    def load_model(cls, reward_output_folder: str) -> Tuple["RewardModel", GPT2Tokenizer]:
        """
        Load a model from disk or from HuggingFace.

        Args:
            reward_output_folder: The folder containing the saved model.

        Returns:
            A tuple containing the loaded model and tokenizer.
        """
        model_save_path = f"{reward_output_folder}/model/model.pth"
        tokenizer_save_path = f"{reward_output_folder}/tokenizer"
        model_name = "gpt2"

        if os.path.exists(model_save_path) and os.path.exists(tokenizer_save_path):
            model = cls.load_model_from_disk(model_save_path, model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_save_path)
        else:
            model = cls(model_name=model_name)
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer


class Generator:
    """
    Model wrapper class to handle training and evaluation of the generator model.
    """

    def __init__(self, config: GeneratorConfig, experiment_name: str, verbose: bool):
        self.name = "Generator"
        self.output_folder = make_output_folder(experiment_name, verbose)
        self.log_file = os.path.join(RESULT_FOLDER, experiment_name, "log.txt")
        self.config = config
        self.verbose = verbose
        self.experiment_name = experiment_name
        self.load_model(experiment_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def log(self, message: str) -> None:
        """
        Logs a message to the specified log file and writes it to stdout if verbose is True.

        Args:
            message (str): The message to log.
        """
        if self.verbose:
            tqdm.write(f"[ {self.name} ] {message}")
        log_to_file(self.log_file, self.name, message)

    def load_model(self, experiment_name: str) -> None:
        """
        Load the model from disk or from HuggingFace.

        Args:
            experiment_name (str): The name of the experiment to load the model from.

        Returns:
            None
        """
        # TODO(dave): create factory for model and tokenizer
        # TODO(dave): refactor path generation
        if os.path.exists(
            os.path.join(RESULT_FOLDER, experiment_name, f"{self.config.model_name}-tokenizer")
        ) and os.path.exists(
            os.path.join(RESULT_FOLDER, experiment_name, f"{self.config.model_name}-model")
        ):
            tokenizer_file = os.path.join(
                RESULT_FOLDER, experiment_name, f"{self.config.model_name}-tokenizer"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_file)
            model_file = os.path.join(
                RESULT_FOLDER, experiment_name, f"{self.config.model_name}-model"
            )
            self.model = AutoModelForCausalLM.from_pretrained(model_file)
            self.log(f"Loaded model {self.config.model_name} from experiment {experiment_name}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.config.model_name)
            self.log(f"Loaded model {self.config.model_name} from HF")
            self.log(" WARNING - you should finetune the model on all papers first!")

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.model.config.eos_token_id

    def train(
        self,
        dataset_config: DatasetConfig,
        training_config: GeneratorTrainingConfig,
    ) -> None:
        """
        Train the generator model from scratch.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset.
            training_config (GeneratorTrainingConfig): The training configuration.

        Returns:
            None
        """
        self.log(
            f"Training model {self.config.model_name} on dataset {dataset_config.generator_dataset_path}"
        )
        model_name = self.config.model_name
        tokenizer = self.tokenizer
        # load model from scratch
        model = GPT2LMHeadModel.from_pretrained(model_name)
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # load datasets
        full_dataset = GeneratorDataset(dataset_config, tokenizer, verbose=self.verbose)
        train_size = int(len(full_dataset) * (1.0 - dataset_config.generator_eval_percentage))
        eval_size = len(full_dataset) - train_size
        train_dataset, eval_dataset = random_split(full_dataset, [train_size, eval_size])
        self.log(f"Dataset split: {train_size} train samples, {eval_size} eval samples")

        # training arguments
        training_args = TrainingArguments(
            output_dir=self.output_folder,
            eval_strategy="epoch",
            logging_steps=training_config.logging_steps,
            learning_rate=training_config.learning_rate,
            per_device_train_batch_size=training_config.per_device_batch_size,
            per_device_eval_batch_size=training_config.per_device_batch_size,
            num_train_epochs=training_config.num_train_epochs,
            weight_decay=training_config.weight_decay,
            warmup_steps=training_config.warmup_step,
            fp16=training_config.fp16,
            push_to_hub=training_config.push_to_hub,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        self.log(f"Perplexity before training: {perplexity}")

        # finetune
        trainer.train()
        trainer.save_model(f"{self.output_folder}/{model_name}-model")
        tokenizer.save_pretrained(f"{self.output_folder}/{model_name}-tokenizer")
        self.log(f"Training complete with {len(train_dataset)} samples")
        self.log(f"Model and tokenizer saved to {self.output_folder}")

        # evaluate perplexity
        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        self.log(f"Perplexity after training: {perplexity}")

    def train_reward_model(
        self, training_config: RLHFTrainingConfig, new_papers: List[Paper]
    ) -> nn.Module:
        """
        Train a reward model on newly generated papers.

        Args:
            training_config (RLHFTrainingConfig): The configuration for training the reward model.
            new_papers (List[Paper]): The new papers to train the reward model on.

        Returns:
            nn.Module: The trained reward model.
        """
        reward_output_folder = os.path.join(self.output_folder, "reward_model")
        model, tokenizer = RewardModel.load_model(reward_output_folder)
        self.log("Loaded reward model")

        dataset = RewardDataset(new_papers, tokenizer)

        training_args = TrainingArguments(
            output_dir=reward_output_folder,
            num_train_epochs=training_config.reward_num_epochs,
            per_device_train_batch_size=training_config.reward_per_device_train_batch_size,
            save_steps=training_config.reward_save_steps,
            save_total_limit=training_config.reward_save_total_limit,
            logging_dir=reward_output_folder,
            logging_steps=training_config.reward_logging_steps,
            learning_rate=training_config.reward_learning_rate,
            weight_decay=training_config.reward_weight_decay,
            eval_strategy="no",
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )

        trainer.train()

        # save model and tokenizer
        model_folder = os.path.join(reward_output_folder, "model")
        os.makedirs(model_folder, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_folder, "model.pth"))
        tokenizer.save_pretrained(f"{reward_output_folder}/tokenizer")
        return model

    def load_value_model(
        self,
        training_config: RLHFTrainingConfig,
        dataset_config: DatasetConfig,
        output_folder: str,
    ) -> Tuple[AutoModelForSequenceClassification, str]:
        """
        Load a value model from the output folder, or load a new one from
        the HuggingFace model hub and pretrain it on the given dataset.

        Args:
            training_config (RLHFTrainingConfig): The configuration for training the value model.
            dataset_config (DatasetConfig): The configuration for the dataset to pretrain the value model on.
            output_folder (str): The folder to save the value model to.

        Returns:
            Tuple[AutoModelForSequenceClassification, str]: The loaded value model and its path.
        """
        value_model_path = os.path.join(output_folder, "value_model")
        if not os.path.exists(value_model_path):
            self.log("Loading value model from HF and pretraining it")
            # load value model from HF
            value_model = AutoModelForSequenceClassification.from_pretrained(self.config.model_name)
            value_model.config.pad_token_id = self.tokenizer.pad_token_id
            train_dataset, test_dataset = ClassificationDataset.load_train_eval_split(
                dataset_config, self.tokenizer
            )
            # pretrain value model on dataset since it's loaded from HF
            training_args = TrainingArguments(
                output_dir=value_model_path,
                num_train_epochs=training_config.value_model_num_epochs,
                per_device_train_batch_size=training_config.value_model_per_device_train_batch_size,
                save_steps=training_config.value_model_save_steps,
                save_total_limit=training_config.value_model_save_total_limit,
                logging_dir=output_folder,
                logging_steps=training_config.value_model_logging_steps,
                learning_rate=training_config.value_model_learning_rate,
                weight_decay=training_config.value_model_weight_decay,
                eval_strategy="no",
                report_to="none",
            )

            trainer = Trainer(
                model=value_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                processing_class=self.tokenizer,
            )
            trainer.train()
            value_model.save_pretrained(value_model_path)
        value_model = AutoModelForSequenceClassification.from_pretrained(value_model_path)
        self.log(f"Loaded value model from {value_model_path}")
        return value_model, value_model_path

    def run_ppo(
        self,
        dataset_config: DatasetConfig,
        training_config: RLHFTrainingConfig,
        reward_model: nn.Module,
    ) -> None:
        """
        Run PPO on the model, saving to the output folder.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset to use for PPO.
            training_config (RLHFTrainingConfig): The configuration for the PPO training process.
            reward_model (nn.Module): The reward model to use for PPO.

        Returns:
            None
        """
        output_folder = os.path.join(self.output_folder, "ppo")
        self.log(f"Running PPO on model, saving to {output_folder}")

        # TODO(dave): replace/extend the training dataset with generated titles (copy mechanism from prod)
        train_dataset = GeneratorDataset(dataset_config, self.tokenizer, verbose=self.verbose)

        training_args = PPOConfig(
            output_dir=output_folder,
            num_train_epochs=training_config.ppo_num_epochs,
            per_device_train_batch_size=training_config.ppo_per_device_train_batch_size,
            save_steps=training_config.ppo_save_steps,
            save_total_limit=training_config.ppo_save_total_limit,
            logging_dir=output_folder,
            logging_steps=training_config.ppo_logging_steps,
            learning_rate=training_config.ppo_learning_rate,
            weight_decay=training_config.ppo_weight_decay,
            eval_strategy="no",
            report_to="none",
        )

        ref_model = create_reference_model(self.model)
        value_model, value_model_path = self.load_value_model(
            training_config, dataset_config, output_folder
        )

        trainer = PPOTrainer(
            args=training_args,
            processing_class=self.tokenizer,
            model=self.model,
            ref_model=ref_model,
            reward_model=reward_model,
            value_model=value_model,
            train_dataset=train_dataset,
            peft_config=None,
        )

        trainer.train()

        # save model, value model and tokenizer
        tokenizer_path = os.path.join(
            RESULT_FOLDER, self.experiment_name, f"{self.config.model_name}-tokenizer"
        )
        model_path = os.path.join(
            RESULT_FOLDER, self.experiment_name, f"{self.config.model_name}-model"
        )
        self.log(f"Saving model and tokenizer to {model_path} and {tokenizer_path}")
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        self.log(f"Saving value model to {value_model_path}")
        value_model.save_pretrained(value_model_path)

    def rlhf(
        self,
        dataset_config: DatasetConfig,
        training_config: RLHFTrainingConfig,
        N: int = 2,
    ) -> None:
        """
        Run the entire RLHF process on the model for one iteration.

        Args:
            dataset_config (DatasetConfig): The configuration for the dataset to use for PPO.
            training_config (RLHFTrainingConfig): The configuration for the RLHF training process.
            N (int, optional): The number of titles to generate. Defaults to 2.

        Returns:
            None
        """
        # generate prompts
        self.log(f"Creating {N} titles...")
        generator = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, device=self.device
        )
        prompt = "The next Paper is | "
        result = generator(
            prompt,
            max_length=512,  # TODO(dave): refactor max length
            num_return_sequences=N,
            temperature=training_config.sampling_temperature,  # higher -> more randomness
            top_k=training_config.sampling_top_k,  # limit sampling to the top-k tokens (lower = more focused)
            top_p=training_config.sampling_top_p,  # nucleus sampling
            do_sample=True,  # random sampling
        )

        papers_to_review = [
            Paper(
                title=result_i["generated_text"].split("|")[1],
                abstract=result_i["generated_text"].split("|")[2],
                authors="",  # no authors from generated papers
                label=None,
            )
            for result_i in result
        ]

        # review prompts
        tui = TextAnnotationTUI(papers_to_review)
        tui.run()
        tui.report()
        reviewed_papers = tui.papers

        # train reward
        reward_model = self.train_reward_model(training_config, reviewed_papers)

        # train model with PPO
        self.run_ppo(dataset_config, training_config, reward_model)
