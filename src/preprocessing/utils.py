import pathlib
import numpy as np
import enum
import shutil
from collections import OrderedDict
from typing import List


class Labels(enum.Enum):
    numbers = [str(i) for i in range(0, 10)]
    lowercase_no_diacritics = [str(i) for i in range(10, 36)]
    lowercase = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)]
    uppercase_no_diacritics = [str(i) for i in range(36, 62)]
    uppercase = [str(i) for i in range(36, 62)] + [str(i) for i in range(71, 80)]
    phcd_paper = [str(i) for i in range(0, 90)]  # all characters


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

    print("Total images found: ", len(all_filenames))
    print(
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


char_id_to_class_name = OrderedDict(
    {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
        10: "a",
        11: "b",
        12: "c",
        13: "d",
        14: "e",
        15: "f",
        16: "g",
        17: "h",
        18: "i",
        19: "j",
        20: "k",
        21: "l",
        22: "m",
        23: "n",
        24: "o",
        25: "p",
        26: "q",
        27: "r",
        28: "s",
        29: "t",
        30: "u",
        31: "v",
        32: "w",
        33: "x",
        34: "y",
        35: "z",
        36: "A",
        37: "B",
        38: "C",
        39: "D",
        40: "E",
        41: "F",
        42: "G",
        43: "H",
        44: "I",
        45: "J",
        46: "K",
        47: "L",
        48: "M",
        49: "N",
        50: "O",
        51: "P",
        52: "Q",
        53: "R",
        54: "S",
        55: "T",
        56: "U",
        57: "V",
        58: "W",
        59: "X",
        60: "Y",
        61: "Z",
        # then lowercase letters of the Polish alphabet: ą, ć, ę, ł, ń, ó, ś, ź, ż
        62: "ą",
        63: "ć",
        64: "ę",
        65: "ł",
        66: "ń",
        67: "ó",
        68: "ś",
        69: "ź",
        70: "ż",
        # then uppercase letters of the Polish alphabet: Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż
        71: "Ą",
        72: "Ć",
        73: "Ę",
        74: "Ł",
        75: "Ń",
        76: "Ó",
        77: "Ś",
        78: "Ź",
        79: "Ż",
        # then special characters: + - : ; $ ! ? @
        80: "+",
        81: "-",
        82: ":",
        83: ";",
        84: "$",
        85: "!",
        86: "?",
        87: "@",
        88: ".",
    }
)


def persist_labels(output_path: pathlib.Path):
    """
    Writes class labels to a .npy file as a numpy array
    """
    class_labels = list(char_id_to_class_name.values())
    output_name = str(output_path / "labels.npy")
    print(f"Writing class labels to {output_name}")
    np.save(output_name, class_labels)
    return output_name
