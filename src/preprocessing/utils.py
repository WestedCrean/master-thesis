import pathlib
import os
import numpy as np
import shutil
import random
from loguru import logger
from PIL import Image


def measure_folder_size(path: pathlib.Path) -> int:
    # calculate size of folder in megabytes
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file()) / 1024 / 1024

def save_split_data(data, output_path, label, split) -> str:
    output_path.mkdir(parents=True, exist_ok=True)
    file_path = output_path / split
    file_path.mkdir(parents=True, exist_ok=True)
    file_path = file_path / f"{label}.npz"
    save_path = file_path.resolve()
    print(f"Saving {split} data to {save_path}")
    np.savez_compressed(str(save_path), data)
    return save_path

def create_npy_files(src_path : pathlib.Path, output_path: pathlib.Path, label: str) -> str:
    output_path.mkdir(parents=True, exist_ok=True)
    img_names = [ str(img_name) for img_name in src_path.iterdir() ]
    img_array = np.array([np.array(Image.open(img_name)) for img_name in img_names])
    np.savez_compressed(output_path / f"{label}.npz", img_array)

def unpack_npz_file(path: pathlib.Path) -> np.ndarray:
    path = str(pathlib.Path(path).resolve())
    return np.load(path)["arr_0"]

def train_test_split(data: list, test_ratio: float = 0.2):
    return np.split(np.array(data), [int(len(data) * (1 - test_ratio))])


def create_train_test_sets(
    path: str, target_path: str, class_name: str, test_ratio: float = 0.2
):
    """
    Takes a path to a directory containing images and splits them into train and test sets.

    Example:

    Source path: src_path/0
    Target path: target_path/0/train and target_path/0/test
    """
    current_path = pathlib.Path(path).resolve()
    target_path = pathlib.Path(target_path)
    target_path.mkdir(parents=True, exist_ok=True)

    # read all files in the directory into list
    all_filenames = list(pathlib.Path(current_path).glob("*.*"))

    # shuffle the list
    np.random.shuffle(all_filenames)

    # split the list into train and test
    train_filenames, test_filenames = train_test_split(all_filenames, test_ratio)

    logger.info("Total images found: ", len(all_filenames))
    logger.info(
        f"Will be split using ratio {test_ratio} into \n\t{len(train_filenames)} train and \n\t{len(test_filenames)} test images\n"
    )

    # create train and test directories in target_path
    (target_path / "train" / class_name).mkdir(parents=True, exist_ok=True)
    (target_path / "test" / class_name).mkdir(parents=True, exist_ok=True)
    # copy files to train and test directories
    for name in train_filenames:
        shutil.copy(name, target_path / "train" / class_name / name.name)
    for name in test_filenames:
        shutil.copy(name, target_path / "test" / class_name / name.name)
