#!/usr/bin/env python

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

import click

from config import ConfigFiles
from generator import Generator
from relevance_model import RelevanceModel


@click.group()
def relevance_model_system_cli() -> None:
    """Relevance Model System"""
    pass


@relevance_model_system_cli.group()
@click.argument("experiment_name", type=str)
@click.argument("config_folder", type=str)
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def generator(ctx: click.Context, experiment_name: str, config_folder: str, verbose: bool) -> None:
    """Generator model commands"""
    ctx.ensure_object(dict)
    ctx.obj["experiment_name"] = experiment_name
    ctx.obj["config_folder"] = config_folder
    ctx.obj["verbose"] = verbose


@generator.command(name="finetune")
@click.pass_context
def finetune_generator(ctx: click.Context) -> None:
    """Finetune the generator with positive labels"""
    # retrieve context variables
    verbose: bool = ctx.obj["verbose"]
    experiment_name: str = ctx.obj["experiment_name"]
    config_folder: str = ctx.obj["config_folder"]
    # run finetuning
    configs: ConfigFiles = ConfigFiles(config_folder)
    generator: Generator = Generator(configs.generator_config, experiment_name, verbose)
    generator.train(
        configs.dataset_config,
        configs.generator_training_config,
    )


@generator.command(name="rlhf")
@click.argument("num_papers_to_generate", type=int, default=100)
@click.pass_context
def train_rlhf(ctx: click.Context, num_papers_to_generate: int) -> None:
    """Train generator model with RLHF"""
    # retrieve context variables
    verbose = ctx.obj["verbose"]
    experiment_name = ctx.obj["experiment_name"]
    config_folder = ctx.obj["config_folder"]
    # run rlhf training
    configs = ConfigFiles(config_folder)
    generator = Generator(configs.generator_config, experiment_name, verbose)
    generator.rlhf(
        dataset_config=configs.dataset_config, training_config=configs.rlhf_training_config,
        N=num_papers_to_generate
    )


@relevance_model_system_cli.group()
@click.argument("experiment_name", type=str)
@click.argument("config_folder", type=str)
@click.option("--verbose", "-v", is_flag=True)
@click.pass_context
def classification(
    ctx: click.Context, experiment_name: str, config_folder: str, verbose: bool
) -> None:
    """Classification model commands"""
    ctx.ensure_object(dict)
    ctx.obj["experiment_name"] = experiment_name
    ctx.obj["config_folder"] = config_folder
    ctx.obj["verbose"] = verbose


@classification.command(name="finetune")
@click.pass_context
def finetune_classification(ctx: click.Context):
    """Finetune classification model"""
    # retrieve context variables
    verbose = ctx.obj["verbose"]
    experiment_name = ctx.obj["experiment_name"]
    config_folder = ctx.obj["config_folder"]
    # run classification finetuning
    configs = ConfigFiles(config_folder)
    classifier = RelevanceModel(configs.relevance_model_config, experiment_name, verbose)
    classifier.finetune(configs.dataset_config, configs.relevance_model_training_config)


@classification.command(name="eval")
@click.pass_context
def evaluate_classification(ctx: click.Context):
    """Evaluate classification model"""
    # retrieve context variables
    verbose = ctx.obj["verbose"]
    experiment_name = ctx.obj["experiment_name"]
    config_folder = ctx.obj["config_folder"]
    # run classification evaluation
    configs = ConfigFiles(config_folder)
    classifier = RelevanceModel(configs.relevance_model_config, experiment_name, verbose)
    classifier.evaluate(configs.dataset_config)


if __name__ == "__main__":
    relevance_model_system_cli()
