import pathlib
import os
import numpy as np
import shutil
import random
from loguru import logger

from .utils import create_train_test_sets


def create_lowercase_latin_letters_with_diacritics_dataset():
    """
    Creates a dataset of handwritten latin letters with diacritics from the PHCD dataset.
    Only lowercase letters are included.
    """
    current_path = pathlib.Path(__file__).resolve().parent

    # check if target folder exists
    target_path = pathlib.Path(
        current_path / "../../data/lowercase_latin_letters_with_diacritics"
    )
    target_path.mkdir(parents=True, exist_ok=True)

    # numbers from 10 to 35 and 62 to 70
    classes_dir = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)]
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
    create_lowercase_latin_letters_with_diacritics_dataset()
