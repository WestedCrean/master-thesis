import pathlib
import numpy as np
import shutil
from loguru import logger
from typing import List


def measure_folder_size(path: pathlib.Path) -> int:
    # calculate size of folder in megabytes
    return sum(f.stat().st_size for f in path.glob("**/*") if f.is_file()) / 1024 / 1024


def create_archive(
    src_path: pathlib.Path, output_path: pathlib.Path, label: str
) -> str:
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(output_path / f"{label}", "gztar", src_path)
    return str(output_path / f"{label}.tar.gz")


def unpack_archive(
    filename: pathlib.Path, output_path: pathlib.Path
) -> List[pathlib.Path]:
    shutil.unpack_archive(filename, output_path, format="gztar")
    return list(output_path.iterdir())


def save_split_data(
    split_paths: List[pathlib.Path], output_path: pathlib.Path, label, split
) -> str:
    output_path = output_path / split / label
    output_path.mkdir(parents=True, exist_ok=True)
    output_path.resolve()

    for path in split_paths:
        shutil.copy(path, output_path)

    return output_path  # shutil.make_archive(output_path, "gztar", output_path)


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
