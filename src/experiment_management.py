import click
from pathlib import Path
import wandb
from preprocessing import upload_raw_dataset, upload_dataset_splits
from loguru import logger

PROJECT_NAME = "master-thesis"


@click.group()
def cli():
    pass


def get_all_wandb_runs(project_name: str):
    """
    Returns all wandb runs names
    """
    api = wandb.Api()
    runs = api.runs(project_name)
    return runs


@cli.command()
def delete_run_data():
    """
    Deletes all runs in wandb master_thesis project
    """
    print("Deleting run data...")
    runs = get_all_wandb_runs(PROJECT_NAME)
    for run in runs:
        print(f"Deleting run {run.name}...")
        try:
            run.delete()
        except Exception:
            print(f"Error deleting run {run.name}")


@cli.command()
def delete_artifacts():
    """
    Deletes all artifacts in wandb master_thesis project
    """
    print("Deleting artifacts...")

    api = wandb.Api()
    # artifacts = api.artifacts(PROJECT_NAME)

    """
    for artifact in artifacts:
        print(f"Deleting artifact {artifact.name}...")
        try:
            artifact.delete()
        except Exception:
            print(f"Error deleting artifact {artifact.name}")
    """


@cli.command()
def create_training_data():
    """
    Uploads raw data & splits for training to wandb
    master_thesis project
    """
    print("Creating training data...")
    upload_raw_dataset()
    upload_dataset_splits()


if __name__ == "__main__":
    cli()
