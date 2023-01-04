import os
import sys
from random import shuffle

import pathlib
import numpy as np
from loguru import logger
import wandb
from preprocessing.utils import create_train_test_sets

def data_split():
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    classes_dir = [str(i) for i in range(10, 80)]
    run = wandb.init(project="master-thesis", job_type="upload")

    data_at = run.use_artifact('raw-letters:latest')
    data_dir = data_at.download()

    data_split_at = wandb.Artifact("split-80-10-10", type="train_val_test_split")
    # create a table with columns we want to track/compare
    preview_dt = wandb.Table(columns=["id", "image", "label", "split"])
    
    for l in labels:
        imgs_per_label = os.path.join(raw_data_source, l)
        if os.path.isdir(imgs_per_label):
            # filter out "DS_Store"
            imgs = [i for i in os.listdir(imgs_per_label) if not i.startswith(".DS")]
            # randomize the order
            shuffle(imgs)
            print(f"Number of images for label {l}: {len(imgs)}")
            img_file_ids = imgs
            for f in img_file_ids:
                file_path = os.path.join(raw_data_source, l, f)
                print(f"File path: {file_path}")
                # add file to artifact by full path
                data_at.add_file(file_path, name=l + "/" + f)

    print("Uploading artifact")
    run.log_artifact(data_at)
    run.finish()

if __name__ == "__main__":
    data_split()
