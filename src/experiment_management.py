import click
import wandb
from preprocessing.upload_raw_dataset import upload_raw_dataset, Labels
from preprocessing.upload_dataset_splits import upload_dataset_splits

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
@click.option("--all", is_flag=True)
@click.option(
    "--label-type",
    type=click.Choice(
        [
            "numbers",
            "lowercase",
            "lowercase_no_diacritics",
            "uppercase",
            "uppercase_no_diacritics",
            "phcd_paper",
        ]
    ),
    required=True,
)
def create_training_data(label_type: str, all: bool):
    """
    Uploads raw data & splits for training to wandb
    master_thesis project
    """
    print("Creating training data...")

    labels_to_process = [label_type]
    if all:
        labels_to_process = [
            "numbers",
            "lowercase",
            "lowercase_no_diacritics",
            "uppercase",
            "uppercase_no_diacritics",
            "phcd_paper",
        ]
    for label in labels_to_process:
        print(f"Using label type: {label}")
        upload_raw_dataset(label_type=Labels[label])
        # upload_dataset_splits()


if __name__ == "__main__":
    cli()
