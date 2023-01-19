import os
import sys
import enum
import pathlib
import wandb

# add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.utils import measure_folder_size, create_archive, persist_labels


class Labels(enum.Enum):
    numbers = [str(i) for i in range(0, 10)]
    lowercase_no_diacritics = [str(i) for i in range(10, 36)]
    lowercase = [str(i) for i in range(10, 36)] + [str(i) for i in range(62, 71)]
    uppercase_no_diacritics = [str(i) for i in range(36, 62)]
    uppercase = [str(i) for i in range(36, 62)] + [str(i) for i in range(71, 80)]
    phcd_paper = [str(i) for i in range(0, 90)]


def upload_raw_dataset(label_type: Labels = Labels.lowercase):
    base_dir = pathlib.Path().resolve().parent
    raw_data_source = f"{base_dir}/data/all_characters"

    labels = label_type.value
    name = label_type.name

    run = wandb.init(project="master-thesis", job_type="upload")

    data_at = wandb.Artifact(name, type=f"raw_data")
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

    # log labels

    label_artifact = wandb.Artifact(f"{name}_labels", type="labels")
    label_file = persist_labels(output_path)
    label_artifact.add_file(label_file, name="labels.npy")
    run.log_artifact(label_artifact)

    run.finish()


if __name__ == "__main__":
    upload_raw_dataset()
