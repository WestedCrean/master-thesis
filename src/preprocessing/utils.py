import pathlib
import os
import numpy as np
import shutil
import random

def train_test_split(data: list, test_ratio: float = 0.2):
    return np.split(np.array(data), [int(len(data) * (1 - test_ratio))])

def create_train_test_sets(path: str, target_path: str, test_ratio: float = 0.2):
    '''
    Takes a path to a directory containing images and splits them into train and test sets.

    Example:

    Source path: src_path/0
    Target path: target_path/0/train and target_path/0/test
    '''
    current_path = pathlib.Path(path).resolve()
    target_path = pathlib.Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    # read all files in the directory into list
    all_filenames = list(pathlib.Path(current_path).glob("*.*"))

    # shuffle the list
    np.random.shuffle(all_filenames)

    # split the list into train and test
    train_filenames, test_filenames = train_test_split(all_filenames, test_ratio)

    print("Total images found: ", len(all_filenames))
    print(f"Will be split using ratio {test_ratio} into \n\t{len(train_filenames)} train and \n\t{len(test_filenames)} test images\n")


    # create train and test directories in target_path
    (target_path / "train").mkdir(parents=True, exist_ok=True)
    (target_path / "test").mkdir(parents=True, exist_ok=True)
    # copy files to train and test directories
    for name in train_filenames:
        shutil.copy(name, target_path / "train" / name.name)
    for name in test_filenames:
        shutil.copy(name, target_path / "test" / name.name)