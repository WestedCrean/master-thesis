import pathlib
import os
import numpy as np
import shutil
import random

from .utils import create_train_test_sets


def create_numbers_dataset():
    """Creates a dataset of handwritten numbers from the PHCD dataset"""
    current_path = pathlib.Path(__file__).resolve().parent

    # check if ../../data/numbers exists
    target_path = pathlib.Path(current_path / "../../data/numbers")
    target_path.mkdir(parents=True, exist_ok=True)

    classes_dir = [str(i) for i in range(10)]
    test_ratio = 0.1

    for i in classes_dir:
        logger.info("Creating train & test set for class: {}".format(i))
        create_train_test_sets(
            current_path / "../../data/all_characters" / i,
            target_path,
            class_name=i,
            test_ratio=test_ratio,
        )
    logger.info("Done")


if __name__ == "__main__":
    create_numbers_dataset()
