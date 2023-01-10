import os
import sys
import enum
import pathlib
import wandb

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.utils import measure_folder_size, create_archive


class Labels(enum.Enum):
    lowercase = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)]
    phcd_paper = [str(i) for i in range(0, 90)]


def upload_raw_dataset(label_type: Labels = Labels.lowercase):
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    labels = label_type.value
    run = wandb.init(project="master-thesis", job_type="upload")

    data_at = wandb.Artifact(f"letters", type=f"raw_data")
    output_path = None
    for l in labels:
        print(f"Processing label {l}")
        label_path = pathlib.Path(raw_data_source) / l
        output_path = pathlib.Path(raw_data_source).parent / "upload"
        print(f"Output path: {output_path.resolve()}")
        create_archive(label_path, output_path, label=l)
        data_at.add_file(output_path / f"{l}.tar.gz", name=f"{l}.tar.gz")

    if output_path:
        print(f"Output files size: {measure_folder_size(output_path)} MB")

    print("Uploading raw artifact")
    run.log_artifact(data_at)
    run.finish()


if __name__ == "__main__":
    upload_raw_dataset()
