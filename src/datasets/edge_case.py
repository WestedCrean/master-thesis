import pathlib
import tensorflow as tf


def edge_case(
    path="../data/edge_case/train",
    subset="training",
    validation_split=0.2,
    batch_size=32,
) -> tf.data.Dataset:
    data_path = pathlib.Path(path)

    return (
        tf.keras.preprocessing.image_dataset_from_directory(
            data_path,
            validation_split=validation_split,
            subset=subset,
            seed=42,
            image_size=(32, 32),
            batch_size=32,
            label_mode="categorical",
        )
        .prefetch(tf.data.experimental.AUTOTUNE)
        .cache()
    )
