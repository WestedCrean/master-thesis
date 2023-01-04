import os
import sys
from random import shuffle

import pathlib
import numpy as np
from loguru import logger
import wandb

def upload_raw_dataset():
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    #labels = [str(i) for i in range(10, 80)]
    labels = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)] # only lowercase
    run = wandb.init(project="master-thesis", job_type="upload")
    
    data_at = wandb.Artifact(f"raw-letters", type=f"raw_data")
    
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
                # add file to artifact by full path
                data_at.add_file(file_path, name=l + "/" + f)

    print("Uploading artifact")
    run.log_artifact(data_at)
    run.finish()

'''
def upload_raw_dataset():
    """
    Uploads a dataset of handwritten latin letters with diacritics from the PHCD dataset to Weights & Biases.
    Both lowercase and uppercase letters are included.
    """

    project_path = pathlib.Path().resolve().parent
    #labels = [str(i) for i in range(10, 80)]
    labels = [str(i) for i in range(0, 9)]
    
    run = wandb.init(project="master-thesis", job_type="upload")
    #data_at = wandb.Artifact(f"latin_letters_with_diacritics", type=f"raw_data")
    data_at = wandb.Artifact(f"numbers", type=f"raw_data")

    for l in labels:
        source_path = pathlib.Path(
            project_path / "data/all_characters" / l,
        )
        imgs = [i for i in source_path.iterdir()]
        np.random.shuffle(imgs)
        for img_path in imgs:
            img_name = img_path.name
            data_at.add_file(str(img_path), name=l + "/" + img_name)

    run.log_artifact(data_at)
    run.finish()
    logger.info("Done") 
'''

if __name__ == "__main__":
    upload_raw_dataset()
