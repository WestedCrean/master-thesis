import pathlib
import tensorflow as tf


def latin_letters(
    path="../data/latin_letters/train",
    subset="training",
    validation_split=0.2,
    batch_size=32,
) -> tf.data.Dataset:
    data_path = pathlib.Path(path)

    return tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset=subset,
        seed=42,
        image_size=(32, 32),
        batch_size=32,
        label_mode="categorical",
    )
