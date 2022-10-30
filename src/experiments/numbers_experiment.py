import wandb
from wandb.keras import WandbCallback
import tensorflow as tf
import numpy as np

from src.datasets import numbers

from src.training.engine import train, test
from src.training.create_models import get_models_for_experiment


def run():
    wandb_project = "phcd_numbers"
    train_data = numbers(subset="training")
    test_data = numbers(subset="validation")

    for model, config in get_models_for_experiment():
        print(config)
        # wandb.init(project=wandb_project, config=config)
        history = train(
            train_data,
            model,
            epochs=10,
            callbacks=[WandbCallback()],
        )
        accuracy, y_true, y_pred = test(test_data, model)
        print({"accuracy": accuracy})
        # wandb.log({"accuracy": accuracy})
        # wandb.sklearn.plot_confusion_matrix(y_true, y_pred)
        # wandb.finish()


if __name__ == "__main__":
    run()
