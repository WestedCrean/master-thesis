from collections import OrderedDict
import wandb
import tensorflow as tf
import pathlib
from loguru import logger

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


def get_class_name(char_id: str):
    return char_id_to_class_name[int(char_id)]


def log_dataset_statistics(
    train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, class_labels: list
):
    """Logs dataset statistics to wandb.

    Args:
        train_dataset (tf.data.Dataset): training dataset
        test_dataset (tf.data.Dataset): test dataset
        class_labels (list): list of class labels
    """

    # create train_labels and test_labels as python lists of class labels
    train_labels = []
    test_labels = []

    for _, labels in train_dataset:
        train_labels.append([c for c in labels.numpy()])

    for _, labels in test_dataset:
        test_labels.append([c for c in labels.numpy()])

    # transform train_labels and test_labels from one-hot encoded to class labels
    train_labels = [
        [class_labels[c] for c in tf.argmax(label, axis=1).numpy()]
        for label in train_labels
    ]

    test_labels = [
        [class_labels[c] for c in tf.argmax(label, axis=1).numpy()]
        for label in test_labels
    ]

    # flatten train_labels and test_labels
    train_labels = [c for label in train_labels for c in label]
    test_labels = [c for label in test_labels for c in label]

    wandb.sklearn.plot_class_proportions(train_labels, test_labels, class_labels)


def log_dataset_to_wandb(
    ds_name: str,
    dataset_path: str,
    wandb_project: str,
):
    """
    Reads dataset and logs it to wandb

    Args:
        ds_name (str): name of the dataset
        train_dataset (tf.data.Dataset): training dataset
        test_dataset (tf.data.Dataset): test dataset
        class_labels (list): list of class labels
        wandb_project (str): name of the wandb project
    """
    logger.info(f"Logging dataset {ds_name} to wandb project {wandb_project}")
    with wandb.init(entity="gratkadlafana", project="numbers") as run:

        dataset_artifact = wandb.Artifact(ds_name, type="raw_data")

        for subdir in ["train", "test"]:
            # count files in each subdirectory
            root_path = pathlib.Path(dataset_path, subdir).resolve()

            # for each directory in the root_path
            for dir in root_path.iterdir():
                # count files in each subdirectory
                files = list(dir.glob("*.png"))
                class_name = get_class_name(dir.name)
                logger.info(f"Found {len(files)} files in subdirectory {class_name}")

                image = wandb.Image(files, caption="Input image")

                dataset_artifact.add_dir(str(dir) + "/", name=class_name)

        run.log_artifact(dataset_artifact)

    logger.info("Done")
