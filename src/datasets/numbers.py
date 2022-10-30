import pathlib
import tensorflow as tf


def numbers(subset='training', validation_split=0.2, batch_size=32) -> tf.data.Dataset:
    data_path = pathlib.Path("../data/numbers/train")

    return tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset=subset,
        seed=42,
        image_size=(32,32),
        batch_size=32
    )

def all_characters(subset='training', validation_split=0.2, batch_size=32) -> tf.data.Dataset:
    data_path = pathlib.Path("../data/all_characters/train")

    return tf.keras.preprocessing.image_dataset_from_directory(
        data_path,
        validation_split=validation_split,
        subset=subset,
        seed=42,
        image_size=(32,32),
        batch_size=32
    )