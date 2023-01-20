import os
import sys
from random import shuffle

import pathlib
import numpy as np
import tensorflow as tf
import wandb

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.utils import (
    measure_folder_size,
    create_archive,
    unpack_archive,
    load_split_image_data,
    save_split_data,
    persist_labels,
    Labels,
)


def upload_dataset_splits(label_type: Labels = Labels.lowercase):
    base_dir = pathlib.Path().resolve().parent
    run = wandb.init(project="master-thesis", job_type="data_split")

    dataset_name = label_type.name

    artifact = run.use_artifact(f"{dataset_name}:latest")
    artifact_path = artifact.download()

    class_archive = [str(l) for l in pathlib.Path(artifact_path).iterdir()]
    print(f"Found {len(class_archive)} classes")
    output_path = (
        pathlib.Path(artifact_path).parent / dataset_name / "splits"
    ).resolve()
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
        f"{dataset_name}_splits",
        type="transformed_data",
        description=f"{dataset_name.capitalize()} dataset split into train, test and val",
    )

    for split in ["train", "test", "val"]:
        split_save_path = (output_path / split).resolve()
        archive_file = create_archive(split_save_path, output_path, label=split)
        split_artifact.add_file(archive_file, name=f"{split}.tar.gz")

    print(f"Output files size: {measure_folder_size(output_path)} MB")
    print("Uploading split artifacts")
    run.log_artifact(split_artifact)
    run.finish()


def upload_tfds(label_type: Labels = Labels.lowercase):
    with wandb.init(project="master-thesis", job_type="data_split") as run:
        split_paths = load_split_image_data(
            run=run, artifact_name=f"{label_type.name}_splits"
        )

        ds_train = tf.keras.utils.image_dataset_from_directory(
            split_paths[0],
            image_size=(32, 32),
            color_mode="grayscale",
        )

        ds_test = tf.keras.utils.image_dataset_from_directory(
            split_paths[1],
            image_size=(32, 32),
            color_mode="grayscale",
        )

        ds_val = tf.keras.utils.image_dataset_from_directory(
            split_paths[2],
            image_size=(32, 32),
            color_mode="grayscale",
        )

        # save datasets on disk then upload to wandb as artifacts

        output_dir = pathlib.Path("./datasets").resolve()
        output_dir.mkdir(exist_ok=True)

        ds_train.save(str(output_dir / "train"), compression="GZIP")
        ds_val.save(str(output_dir / "val"), compression="GZIP")
        ds_test.save(str(output_dir / "test"), compression="GZIP")

        artifact = wandb.Artifact(
            f"{label_type.name}_splits_tfds",
            type="dataset",
            description="Dataset splits in tf.data.Dataset format",
        )
        artifact.add_dir(output_dir)
        run.log_artifact(artifact)


if __name__ == "__main__":
    upload_dataset_splits()
