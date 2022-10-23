import tensorflow as tf
import numpy as np
import wandb 

def train(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    epochs: int,
    callbacks: list = [],
):
    return model.fit(
        dataset,
        epochs=epochs,
        callbacks=callbacks,
    )

def test(
    dataset: tf.data.Dataset,
    model: tf.keras.Model,
    class_names: list,
):
    loss, accuracy = model.evaluate(dataset)
    print(f"Accuracy: {accuracy:.4f}, loss: {loss:.4f}")
    
    # generate y_true and y_pred
    y_true = []
    y_pred = []
    for x, y in dataset:
        y_true.append(np.argmax(y, axis=1))
        y_pred.append(np.argmax(model.predict(x), axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # return the accuracy and wandb.plot.confusion_matrix
    return (
        accuracy,
        wandb.plot.confusion_matrix(probs=None, y_true=np.argmax(y_true, axis=1), preds=y_pred, class_names=class_names),
    )

