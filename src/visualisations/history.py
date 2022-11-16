import wandb


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
