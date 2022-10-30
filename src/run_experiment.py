import click
from loguru import logger
import warnings
from pathlib import Path


@click.command()
@click.argument("experiment_name", type=str)
def main(experiment_name):
    """Runs experiment given by experiment_name"""
    logger.info("running experiment %s", experiment_name)

    if experiment_name == "numbers":
        from experiments.numbers_experiment import run

        run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
