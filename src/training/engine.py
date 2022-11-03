import tensorflow as tf
import numpy as np
from loguru import logger


def train(
    train_dataset: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int,
    validation_dataset: tf.data.Dataset = None,
    callbacks: list = [],
):
    return model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )


def test(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
):
    loss, accuracy = model.evaluate(dataset)
    logger.info(f"Accuracy: {accuracy:.4f}, loss: {loss:.4f}")

    # generate y_true and y_pred
    y_true = []
    y_pred = []
    y_probas = []
    for x, y in dataset:
        y_true.append(np.argmax(y, axis=1))
        y_pred.append(np.argmax(model.predict(x), axis=1))
        y_probas.append(model.predict(x))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_probas = np.concatenate(y_probas)

    # return the accuracy and wandb.plot.confusion_matrix
    return (accuracy, y_true, y_pred, y_probas)
