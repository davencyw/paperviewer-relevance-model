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

from dataclasses import dataclass

from dataset import DatasetConfig
from generator import (GeneratorConfig, GeneratorTrainingConfig,
                       RLHFTrainingConfig)
from relevance_model import RelevanceModelConfig, RelevanceModelTrainingConfig


@dataclass
class ConfigFiles:
    """
    Collection of all configuration files for the whole system.
    """

    generator_config: GeneratorConfig
    generator_training_config: GeneratorTrainingConfig
    relevance_model_config: RelevanceModelConfig
    relevance_model_training_config: RelevanceModelTrainingConfig
    rlhf_training_config: RLHFTrainingConfig
    dataset_config: DatasetConfig

    def __init__(self, config_folder: str):
        self.generator_config = GeneratorConfig.from_json(
            f"{config_folder}/generator_model_config.json"
        )
        self.generator_training_config = GeneratorTrainingConfig.from_json(
            f"{config_folder}/generator_training_config.json"
        )
        self.relevance_model_config = RelevanceModelConfig.from_json(
            f"{config_folder}/relevance_model_config.json"
        )
        self.relevance_model_training_config = RelevanceModelTrainingConfig.from_json(
            f"{config_folder}/relevance_model_training_config.json"
        )
        self.rlhf_training_config = RLHFTrainingConfig.from_json(
            f"{config_folder}/rlhf_training_config.json"
        )

        # NOTE(dave): swap this out for the database config (only for initial finetuning the generator)
        self.dataset_config = DatasetConfig.from_json(f"{config_folder}/dataset_db_config.json")
        # self.dataset_config = DatasetConfig.from_json(f"{config_folder}/dataset_config.json")
