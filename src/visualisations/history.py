import wandb
import tensorflow as tf


def plot_history(history):
    data = [
        [epoch, acc, val_acc]
        for (epoch, acc, val_acc) in zip(
            history.history["epoch"],
            history.history["categorical_accuracy"],
            history.history["val_categorical_accuracy"],
        )
    ]
    accuracy_history = wandb.Table(
        data=data, columns=["epoch", "categorical_accuracy", "val_categorical_accuracy"]
    )
    return wandb.plot.line(
        accuracy_history, "epoch", "val_categorical_accuracy", title="Accuracy History"
    )


def plot_loss_history(history: tf.keras.callbacks.History):
    # get step, loss, val_loss
    data = [
        [epoch, loss, val_loss]
        for (epoch, loss, val_loss) in zip(
            history.history["epoch"],
            history.history["loss"],
            history.history["val_loss"],
        )
    ]
    loss_history = wandb.Table(data=data, columns=["epoch", "loss", "val_loss"])
    # plot 2 lines on the same graph - loss and val_loss
    return wandb.plot.line(
        loss_history,
        "epoch",
        ["loss", "val_loss"],
        title="Loss History",
    )


def plot_accuracy_history(history: tf.keras.callbacks.History):
    # get step, categorical_accuracy, val_categorical_accuracy
    data = [
        [epoch, acc, val_acc]
        for (epoch, acc, val_acc) in zip(
            history.history["epoch"],
            history.history["categorical_accuracy"],
            history.history["val_categorical_accuracy"],
        )
    ]
    accuracy_history = wandb.Table(
        data=data, columns=["epoch", "categorical_accuracy", "val_categorical_accuracy"]
    )
    # plot 2 lines on the same graph - categorical_accuracy and val_categorical_accuracy
    return wandb.plot.line(
        accuracy_history,
        "epoch",
        ["categorical_accuracy", "val_categorical_accuracy"],
        title="Accuracy History",
    )
