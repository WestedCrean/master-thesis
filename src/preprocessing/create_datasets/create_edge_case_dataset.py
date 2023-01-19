import pathlib
import os
import numpy as np
import shutil
import random


from .utils import create_train_test_sets


def create_edge_case_dataset():
    """
    Creates a dataset of handwritten letters from the PHCD dataset.
    Only latin letters with diacritics and their non-diacritics counterparts are included.
    """
    current_path = pathlib.Path(__file__).resolve().parent

    # check if target folder exists
    target_path = pathlib.Path(current_path / "../../data/edge_case")
    target_path.mkdir(parents=True, exist_ok=True)

    classes_dir = [str(i) for i in range(10)]
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
    create_edge_case_dataset()
