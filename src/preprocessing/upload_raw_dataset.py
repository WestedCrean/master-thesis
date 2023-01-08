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
from utils import measure_folder_size, create_npy_files
        
def upload_raw_dataset():
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    labels = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)] # only lowercase
    run = wandb.init(project="master-thesis", job_type="upload")
    
    data_at = wandb.Artifact(f"letters", type=f"raw_data")
    output_path = None
    for l in labels:
        print(f"Processing label {l}")
        label_path = pathlib.Path(raw_data_source) / l
        output_path = pathlib.Path(raw_data_source).parent / "npy"
        create_npy_files(label_path, output_path, label=l)
        data_at.add_file(output_path / f"{l}.npz", name=f"{l}.npz")

    if output_path:
        print(f"Output files size: {measure_folder_size(output_path)} MB")

    print("Uploading raw artifact")
    run.log_artifact(data_at)

    label_artifact = wandb.Artifact(f"letters_labels", type="labels")
    label_file = persist_labels(output_path)
    label_artifact.add_file(label_file, name="labels.npy")

    run.log_artifact(label_artifact)
    run.finish()

if __name__ == "__main__":
    upload_raw_dataset()
