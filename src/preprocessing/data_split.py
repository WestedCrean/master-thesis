import os
import sys
from random import shuffle

import pathlib
import numpy as np

import wandb
from preprocessing.utils import create_train_test_sets


def train_test_split(data: list, test_ratio: float = 0.2):
    return np.split(np.array(data), [int(len(data) * (1 - test_ratio))])


def log_split_to_wandb(split: str, img_filenames: list):
    run = wandb.init(
        project="master-thesis", job_type="data_split", entity=f"{split}_upload"
    )
    data_at = wandb.Artifact(split, type=f"splits")
    for l in os.listdir(data_dir):
        source_path = pathlib.Path(
            data_dir / l,
        )
        imgs = [i for i in source_path.iterdir()]
        np.random.shuffle(imgs)
        for img_path in imgs:
            img_name = img_path.name
            data_at.add_file(img_path, name=l + "/" + img_name)

    run.log_artifact(data_at)
    run.finish()


def data_split():
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    classes_dir = [str(i) for i in range(10, 80)]

    data_at = run.use_artifact("raw-letters:latest")
    data_dir = data_at.download()

    data_split_at = wandb.Artifact("split-80-10-10", type="data_splits")
    # create a table with columns we want to track/compare
    # preview_dt = wandb.Table(columns=["id", "image", "label", "split"])

    for l in labels:

        imgs_per_label = os.path.join(data_dir, l)
        class_images = os.listdir(imgs_per_label)
        total_images_in_class = len(class_images)

        # split the list into train, val and test sets using 80:10:10 ratio

        train_filenames, test_filenames = train_test_split(class_images, test_ratio=0.2)
        train_filenames, val_filenames = train_test_split(
            train_filenames, test_ratio=0.125
        )

        print(f"Number of images for label {l}: total {total_images_in_class}")
        print(f"in train set: {len(train_filenames)}")
        print(f"in val set: {len(val_filenames)}")
        print(f"in test set: {len(test_filenames)}")

        # log the splits to wandb
        log_split_to_wandb("train", train_filenames)
        log_split_to_wandb("val", val_filenames)
        log_split_to_wandb("test", test_filenames)
    print("Done")


if __name__ == "__main__":
    data_split()
