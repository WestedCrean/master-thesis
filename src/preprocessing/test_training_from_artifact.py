import wandb
import tensorflow as tf
import numpy as np
import pathlib
from typing import List


def load_data(split: str = "train", labels: List[str]) -> np.ndarray:
    """
    Loads data from an artifact and returns it as a numpy array.
    """
    if split not in ["train", "test", "val"]:
        raise ValueError("Split must be either train, test or val")

    artifact_name = f"letters_{split}"
    run = wandb.init(project="master-thesis", job_type="preprocessing")
    artifact = run.use_artifact(f"master-thesis/{artifact_name}:latest")
    artifact_dir = artifact.download()
    data = []
    for label in labels:
        data.append(
            np.load(
                pathlib.Path(artifact_dir) / f"{label}.npz", allow_pickle=True
            )["arr_0"]
        )
    data = np.concatenate(data)
    return data