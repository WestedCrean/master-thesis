import pathlib
import shutil

current_path = pathlib.Path(__file__).resolve().parent

# check if ../../data/numbers exists
target_path = pathlib.Path(current_path / "../../data/numbers")
target_path.mkdir(parents=True, exist_ok=True)

for i in range(10):
    pathlib.Path(current_path / "../../data/numbers/" / str(i)).mkdir(
        parents=True, exist_ok=True
    )
    for file in pathlib.Path(current_path / "../../data/all_characters" / str(i)).glob(
        "*.*"
    ):
        shutil.copy(file, target_path / str(i) / file.name)
