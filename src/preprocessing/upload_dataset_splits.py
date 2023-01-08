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
from utils import measure_folder_size, create_npy_files, unpack_npz_file, save_split_data
        
def upload_dataset_splits():
    base_dir = pathlib.Path().resolve().parent
    #raw_data_source = f"{base_dir}/data/all_characters"
    run = wandb.init(project="master-thesis", job_type="data_split")
    
    artifact = run.use_artifact("letters:latest")
    artifact_path = artifact.download()

    splits_artifacts = {}

    for split in ["train", "test", "val"]:
        split_artifact = wandb.Artifact(
            f"letters_{split}", type="dataset", description=f"Letters {split} dataset"
        )
        splits_artifacts[split] = split_artifact
    
    files = [str(l) for l in pathlib.Path(artifact_path).iterdir()]
    output_path = None

    for f in files:
        label = f.replace(".npz", "").split("/")[-1]
        file_path = pathlib.Path(f).resolve()
        print(f"Processing label {label}")
        data = unpack_npz_file(file_path)
        output_path = (pathlib.Path(artifact_path).parent / "splits").resolve()
        print(f"Output path: {output_path}")
        shuffle(data)

        for split, split_ratio in zip(["train", "test", "val"], [0.8, 0.1, 0.1]):
            # split data into train, test and val with 80%, 10% and 10% respectively
            split_data = data[:int(len(data) * split_ratio)]
            split_save_path = save_split_data(split_data, output_path, label, split)
            splits_artifacts[split].add_file(split_save_path, name=f"{label}.npz")

    print(f"Output files size: {measure_folder_size(output_path)} MB")
    print("Uploading split artifacts")
    for split, split_artifact in splits_artifacts.items():
        run.log_artifact(split_artifact)
    run.finish()

if __name__ == "__main__":
    upload_dataset_splits()
