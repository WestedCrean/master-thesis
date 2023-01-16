import os
import warnings
import click
import wandb
from models import baseline_phcd, resnet, efficientnet
from training.sweep import launch_sweep, base_sweep_config


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--model",
    type=click.Choice(["baseline", "resnet", "efficientnet"], case_sensitive=False),
    required=True,
)
def run(model):
    train_fn = None
    defaults = None
    if model == "baseline":
        defaults = baseline_phcd.get_defaults()
        train_fn = baseline_phcd.train

    elif model == "resnet":
        defaults = resnet.get_defaults()
        train_fn = resnet.train

    elif model == "efficientnet":
        defaults = efficientnet.get_defaults()
        train_fn = efficientnet.train

    train_fn(config=defaults, job_type="training")


@main.command()
@click.option(
    "--model",
    type=click.Choice(["baseline", "resnet", "efficientnet"], case_sensitive=False),
    required=True,
)
def run_sweep(model):
    """
    Runs a wandb sweep
    """
    train_fn = None
    current_sweep_config = None
    if model == "baseline":
        current_sweep_config = baseline_phcd.get_sweep_params()
        train_fn = baseline_phcd.train

    elif model == "resnet":
        current_sweep_config = resnet.get_sweep_params()
        train_fn = resnet.train

    elif model == "efficientnet":
        current_sweep_config = efficientnet.get_sweep_params()
        train_fn = efficientnet.train

    launch_sweep(train_fn, number_of_runs=20, sweep_config=current_sweep_config)


if __name__ == "__main__":
    # disable tensorflow warnings and info
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
