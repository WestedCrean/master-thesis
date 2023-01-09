import os
import sys
from random import shuffle

import pathlib
import numpy as np
from loguru import logger
import wandb

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.utils import persist_labels
from preprocessing.utils import (
    measure_folder_size,
    create_archive,
    unpack_archive,
    save_split_data,
)


def upload_dataset_splits():
    base_dir = pathlib.Path().resolve().parent
    run = wandb.init(project="master-thesis", job_type="data_split")

    artifact = run.use_artifact("letters:latest")
    artifact_path = artifact.download()

    splits_artifacts = {}

    for split in ["train", "test", "val"]:
        split_artifact = wandb.Artifact(
            f"letters_{split}", type="dataset", description=f"Letters {split} dataset"
        )
        splits_artifacts[split] = split_artifact

    class_archive = [str(l) for l in pathlib.Path(artifact_path).iterdir()]
    print(f"Found {len(class_archive)} classes")
    output_path = (pathlib.Path(artifact_path).parent / "splits").resolve()
    for archive_path in class_archive:
        label = archive_path.replace(".tar.gz", "").split("/")[-1]
        file_path = pathlib.Path(archive_path).resolve()
        print(f"Processing label {label}")
        unpacked_output_path = (pathlib.Path(artifact_path).parent / "raw").resolve()
        file_paths = unpack_archive(file_path, unpacked_output_path / label)
        print(f"Found {len(file_paths)} files in archive")

        for split, split_ratio in zip(["train", "test", "val"], [0.8, 0.1, 0.1]):
            # split data into train, test and val with 80%, 10% and 10% respectively
            split_paths = file_paths[: int(len(file_paths) * split_ratio)]
            split_save_path = save_split_data(split_paths, output_path, label, split)

    print("Creating split artifacts")

    print(f"Output path: {output_path.resolve()}")
    split_artifact = wandb.Artifact(
        f"letters_splits",
        type="transformed_data",
        description=f"Letters dataset split into train, test and val",
    )

    for split in ["train", "test", "val"]:
        split_save_path = (output_path / split).resolve()
        archive_file = create_archive(split_save_path, output_path, label=split)
        split_artifact.add_file(archive_file, name=f"{split}.tar.gz")

    print(f"Output files size: {measure_folder_size(output_path)} MB")
    print("Uploading split artifacts")
    run.log_artifact(split_artifact)
    run.finish()


if __name__ == "__main__":
    upload_dataset_splits()
