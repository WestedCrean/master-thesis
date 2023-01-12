import wandb
import pathlib
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt


def load_data(run: wandb.sdk.wandb_run.Run) -> pathlib.Path:
    """
    Unpacks data from an artifact into a folder and returns the path to the folder.
    """

    artifact_name = f"letters_splits"
    artifact = run.use_artifact(f"master-thesis/{artifact_name}:latest")
    artifact_dir = pathlib.Path(
        f"./artifacts/{artifact.name.replace(':', '-')}"
    ).resolve()
    if not artifact_dir.exists():
        artifact_dir = artifact.download()
        artifact_dir = pathlib.Path(artifact_dir).resolve()
        for split_file in artifact_dir.iterdir():
            if split_file.name.endswith(".tar.gz"):
                split = split_file.name.replace(".tar.gz", "")
                shutil.unpack_archive(split_file, artifact_dir / split, format="gztar")

    return [artifact_dir / split for split in ["train", "test", "val"]]


def get_number_of_classes(ds: tf.data.Dataset) -> int:
    """
    Returns the number of classes in a dataset.
    """
    return len(ds.class_names)


def create_tf_dataset(split_path: pathlib.Path, batch_size: int = 32):
    """
    Creates a tf dataset from path containing a folder for each class.
    """
    ds = tf.keras.utils.image_dataset_from_directory(
        split_path,
        image_size=(32, 32),
        batch_size=batch_size,
        color_mode="grayscale",
    )
    return ds


def preprocess_dataset(ds: tf.data.Dataset, cache: bool = True) -> tf.data.Dataset:
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))  # normalize
    if cache:
        ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def calculate_model_size_on_disk(path: str) -> int:
    return pathlib.Path(path).stat().st_size


def calculate_model_num_parameters(model: tf.keras.Model) -> int:
    return model.count_params()


def calculate_model_flops() -> str:
    pass


def plot_history(history):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    epochs = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs, history.history["accuracy"], label="accuracy")
    ax.plot(epochs, history.history["val_accuracy"], label="val_accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="lower right")

    plt.show()
