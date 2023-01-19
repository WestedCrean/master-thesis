import pathlib
import os
import numpy as np
import shutil
import random


from ..utils import create_train_test_sets


def create_latin_letters_dataset():
    """
    Creates a dataset of handwritten latin letters from the PHCD dataset.
    Both lowercase and uppercase letters are included.
    """
    current_path = pathlib.Path(__file__).resolve().parent

    # check if target folder exists
    target_path = pathlib.Path(current_path / "../../data/latin_letters")
    target_path.mkdir(parents=True, exist_ok=True)

    classes_dir = [str(i) for i in range(10, 62)]
    test_ratio = 0.1

    for i in classes_dir:
        print("Creating train & test set for class: {}".format(i))
        create_train_test_sets(
            current_path / "../../data/all_characters" / i,
            target_path,
            class_name=i,
            test_ratio=test_ratio,
        )
    print("Done")


if __name__ == "__main__":
    create_latin_letters_dataset()
