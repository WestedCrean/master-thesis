import wandb


def plot_history(history):
    data = [
        [x, y]
        for (x, y) in zip(
            history.history["categorical_accuracy"],
            history.history["val_categorical_accuracy"],
        )
    ]
    accuracy_history = wandb.Table(data=data, columns=["x", "y"])
    return wandb.plot.line(accuracy_history, "x", "y", title="Accuracy History")
