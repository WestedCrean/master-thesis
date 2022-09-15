import pathlib
import os
import numpy as np
import shutil
import random

from utils import create_train_test_sets

current_path = pathlib.Path(__file__).resolve().parent

# check if ../../data/numbers exists
target_path = pathlib.Path(current_path / "../../data/numbers")
target_path.mkdir(parents=True, exist_ok=True)

classes_dir = [str(i) for i in range(10)]
test_ratio = 0.2

for i in classes_dir:
    print("Creating train & test set for class: {}".format(i))
    create_train_test_sets(current_path / "../../data/all_characters" / i, target_path, test_ratio)
print("Done")