import os
import click
import logging
from pathlib import Path

import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt
import tensorflow as tf

from training.engine import train, test
from training.create_models import get_models_for_experiment
from datasets import numbers, get_class_name, log_dataset_statistics


def run():
    wandb_project = "phcd_numbers"
    train_data = numbers(subset="training")
    validation_data = numbers(subset="validation")
    test_data = numbers(path="../data/numbers/test", subset=None, validation_split=None)

    class_labels = [get_class_name(cn) for cn in validation_data.class_names]

    with wandb.init(project=wandb_project, config={"class_labels": class_labels}):
        # dataset statistics
        log_dataset_statistics(train_data, validation_data, class_labels)

    for model, config in get_models_for_experiment():
        wandb.init(project=wandb_project, config=config, name=config["model_name"])
        history = train(
            train_data,
            model,
            epochs=10,
            validation_dataset=validation_data,
            callbacks=[
                WandbCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=6, restore_best_weights=True
                ),
            ],
        )

        accuracy, y_true, y_pred, y_probas = test(test_data, model)
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred)
        wandb.sklearn.plot_roc(y_true, y_probas, class_labels)
        wandb.finish()


if __name__ == "__main__":
    run()
