from halo import Halo
import pathlib
import shutil
import sys
import time
import datetime
import numpy as np
import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import matplotlib.pyplot as plt
import zipfile

from typing import List


def load_data(run, artifact_name="phcd_paper_splits_tfds") -> List[tf.data.Dataset]:
    """
    Downloads datasets from a wandb artifact and loads them into a list of tf.data.Datasets.
    """

    artifact = run.use_artifact(f"master-thesis/{artifact_name}:latest")
    artifact_dir = pathlib.Path(
        f"./artifacts/{artifact.name.replace(':', '-')}"
    ).resolve()
    if not artifact_dir.exists():
        artifact_dir = artifact.download()
        artifact_dir = pathlib.Path(artifact_dir).resolve()

    # if tf.__version__ minor is less than 10, use
    # tf.data.experimental.load instead of tf.data.Dataset.load

    if int(tf.__version__.split(".")[1]) < 10:
        load_function = tf.data.experimental.load
    else:
        load_function = tf.data.Dataset.load

    output_list = []
    for split in ["train", "test", "val"]:
        ds = load_function(str(artifact_dir / split), compression="GZIP")
        output_list.append(ds)

    return output_list


def get_readable_class_labels(subset="phcd_paper"):
    if subset == "phcd_paper":
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "ą",
            "ć",
            "ę",
            "ł",
            "ń",
            "ó",
            "ś",
            "ź",
            "ż",
            "Ą",
            "Ć",
            "Ę",
            "Ł",
            "Ń",
            "Ó",
            "Ś",
            "Ź",
            "Ż",
            "+",
            "-",
            ":",
            ";",
            "$",
            "!",
            "?",
            "@",
            ".",
        ]
    elif subset == "uppercase":
        return [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            "Ą",
            "Ć",
            "Ę",
            "Ł",
            "Ń",
            "Ó",
            "Ś",
            "Ź",
            "Ż",
        ]
    elif subset == "lowercase":
        return [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "ą",
            "ć",
            "ę",
            "ł",
            "ń",
            "ó",
            "ś",
            "ź",
            "ż",
        ]
    elif subset == "numbers":
        return ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    elif subset == "uppercase_no_diacritics":
        return [
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
        ]
    elif subset == "lowercase_no_diacritics":
        return [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]


def calculate_accuracy_per_class(model, test_dataset, test_dataset_name):
    """
    Calculates the accuracy per class for a given model and test dataset.

    Returns dict with class labels as keys and accuracy as values.
    """

    y_pred = model.predict(test_dataset)
    y_pred = np.argmax(y_pred, axis=1)
    # get labels
    y_true = test_dataset.map(lambda x, y: y).as_numpy_iterator()
    y_true = np.concatenate(list(y_true))
    # calculate accuracy per class
    labels = get_readable_class_labels(test_dataset_name)
    class_accuracy = np.zeros(len(labels))
    for i, label in enumerate(labels):
        class_accuracy[i] = np.sum(y_pred[y_true == i] == i) / np.sum(y_true == i)
    return {label: acc for label, acc in zip(labels, class_accuracy)}


def plot_accuracy_per_class(class_accuracy_dict):
    plt.figure(figsize=(10, 5))
    labels = list(class_accuracy_dict.keys())
    class_accuracy = list(class_accuracy_dict.values())
    plt.bar(labels, class_accuracy)
    plt.xticks(labels)
    plt.xlabel("Class")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per class")
    plt.show()


def accuracy_table(class_accuracy_dict):
    labels = list(class_accuracy_dict.keys())
    class_accuracy = list(class_accuracy_dict.values())
    return wandb.Table(
        columns=["Class", "Accuracy"], data=list(zip(labels, class_accuracy))
    )


def get_number_of_classes(ds: tf.data.Dataset) -> int:
    """
    Returns the number of classes in a dataset.
    """
    labels_iterator = ds.map(lambda x, y: y).as_numpy_iterator()
    labels = np.concatenate(list(labels_iterator))
    return len(np.unique(labels))


def get_number_of_examples(ds: tf.data.Dataset) -> int:
    """
    Returns the number of examples in a dataset.
    """
    return sum(1 for _ in ds)


def preprocess_dataset(
    ds: tf.data.Dataset, batch_size: int, cache: bool = True
) -> tf.data.Dataset:
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # normalize
    ds = ds.unbatch().batch(batch_size)
    if cache:
        ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def calculate_model_compressed_size_on_disk(path: str) -> int:
    compressed_path = path + ".zip"
    with zipfile.ZipFile(compressed_path, "w", compression=zipfile.ZIP_DEFLATED) as f:
        f.write(path)
    return pathlib.Path(compressed_path).stat().st_size


def prepare_data(model_name, artifact_name, defaults):
    with wandb.init(
        project="master-thesis",
        job_type="training",
        name=model_name,
        config=defaults,
        tags=[artifact_name],
    ) as run:

        # hyperparameters
        epochs = wandb.config.epochs
        bs = wandb.config.batch_size

        ds_train, ds_test, ds_val = load_data(run, artifact_name=artifact_name)

        # num_classes = get_number_of_classes(ds_val)

        ds_train = preprocess_dataset(ds_train, batch_size=bs)
        ds_val = preprocess_dataset(ds_val, batch_size=bs)
        ds_test = preprocess_dataset(ds_test, batch_size=bs, cache=False)

        return ds_train, ds_test, ds_val


def train_model(model, ds_train, ds_val, ds_test):

    return model, history, test_loss, test_acc


def get_model(num_classes):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", input_shape=(32, 32, 1)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.SpatialDropout2D(0.2),
            tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.SpatialDropout2D(0.25),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.SpatialDropout2D(0.3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def __main__(batch_size=32 * 2):

    model_name = f"architecture-7"
    artifact_name = "phcd_paper_splits_tfds"
    defaults = dict(batch_size=batch_size, epochs=50, optimizer="adam")

    with Halo(text="Preparing data...", spinner="dots"):
        ds_train, ds_test, ds_val = prepare_data(model_name, artifact_name, defaults)

    num_classes = get_number_of_classes(ds_val)

    model = get_model(num_classes)

    with Halo(text="Training model...", spinner="dots"):
        t1 = time.time()
        history = model.fit(
            ds_train,
            epochs=wandb.config.epochs,
            validation_data=ds_val,
            callbacks=[
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.2, patience=5, min_lr=0.000001
                ),
            ],
        )
        t2 = time.time()

    with Halo(text="Evaluating model...", spinner="dots"):
        t3 = time.time()
        test_loss, test_acc = model.evaluate(ds_test)
        t4 = time.time()

    print("Training time: ", t2 - t1)
    print("Evaluation time: ", t4 - t3)


if __name__ == "__main__":
    # read batch_size from sys.argv
    try:
        batch_size = int(sys.argv[1])
    except IndexError:
        batch_size = 32 * 2
    __main__(batch_size)
