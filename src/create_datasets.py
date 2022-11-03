import click
from loguru import logger
from pathlib import Path

from preprocessing.create_numbers_dataset import create_numbers_dataset


@click.command()
def main():
    """Creates datasets"""
    logger.info("Creating datasets...")

    create_numbers_dataset()


if __name__ == "__main__":
    main()
