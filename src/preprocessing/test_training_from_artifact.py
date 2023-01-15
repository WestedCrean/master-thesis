import wandb
import tensorflow as tf
import numpy as np
import pathlib
from typing import List


def load_data(run) -> List[tf.data.Dataset]:
    """
    Downloads datasets from a wandb artifact and loads them into a list of tf.data.Datasets.
    """

    artifact_name = f"letters_splits_tfds"
    artifact = run.use_artifact(f"master-thesis/{artifact_name}:latest")
    artifact_dir = pathlib.Path(
        f"./artifacts/{artifact.name.replace(':', '-')}"
    ).resolve()
    if not artifact_dir.exists():
        artifact_dir = artifact.download()
        artifact_dir = pathlib.Path(artifact_dir).resolve()

    output_list = []
    for split in ["train", "test", "val"]:
        ds = tf.data.Dataset.load(str(artifact_dir / split), compression="GZIP")
        output_list.append(ds)

    return output_list
