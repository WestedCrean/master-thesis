import os
import warnings
import click
from loguru import logger
import wandb

from datasets import log_dataset_to_wandb


@click.group()
def main():
    pass


@main.command()
@click.argument("experiment_name", type=str)
def upload_dataset_to_wandb(experiment_name):
    """Runs experiment given by experiment_name"""
    logger.info(f'Uploading dataset from experiment "{experiment_name}"')

    if experiment_name == "numbers":
        log_dataset_to_wandb(
            "numbers_data", f"../data/{experiment_name}", experiment_name
        )

    logger.info(f'Finished experiment "{experiment_name}"')


@main.command()
@click.argument("experiment_name", type=str)
def run_experiment(experiment_name):
    """Runs experiment given by experiment_name"""
    logger.info(f'Running experiment "{experiment_name}"')

    if experiment_name == "numbers":
        from experiments.numbers_experiment import run

        run()

    elif experiment_name == "cnn":
        from experiments.cnn_experiment import run

        run()
    logger.info(f'Finished experiment "{experiment_name}"')


@main.command()
@click.argument("experiment_name", type=str)
def clear_experiment(experiment_name):
    """Deletes all runs from wandb experiments"""
    logger.info(f"Cleaning wandb project {experiment_name}")
    api = wandb.Api()
    runs = api.runs(f"gratkadlafana/phcd_numbers")
    for run in runs:
        run.delete()
    logger.info("Project runs deleted")


if __name__ == "__main__":
    # disable tensorflow warnings and info
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
