import pathlib
import os
import numpy as np
import shutil
import random

from utils import create_train_test_sets

current_path = pathlib.Path(__file__).resolve().parent

# check if ../../data/numbers exists
target_path = pathlib.Path(current_path / "../../data/numbers/train+test")
target_path.mkdir(parents=True, exist_ok=True)

classes_dir = [str(i) for i in range(10)]
test_ratio = 0.2

for cls in classes_dir:
    create_train_test_sets(current_path / "../../data/numbers" / cls, target_path / cls, test_ratio)

for i in classes_dir:
    pathlib.Path(current_path / "../../data/numbers/" / classes_dir).mkdir(
        parents=True, exist_ok=True
    )
    for file in pathlib.Path(current_path / "../../data/all_characters" / classes_dir).glob(
        "*.*"
    ):
        shutil.copy(file, target_path / classes_dir / file.name)
