from pathlib import Path

import wandb
from wandb.keras import WandbCallback

import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger


from training.engine import train, test
from training.create_models import get_models_for_experiment
from datasets import (
    lowercase_latin_letters_with_diacritics,
    get_class_name,
    log_dataset_statistics,
)
from visualisations.history import plot_history
from visualisations.classification_metrics import (
    get_classification_report,
    plot_classification_accuracy,
)


def run(clear_project_before=False):
    wandb_project = "lowercase_latin_letters_with_diacritics"
    train_data = lowercase_latin_letters_with_diacritics(subset="training")
    validation_data = lowercase_latin_letters_with_diacritics(subset="validation")
    test_data = lowercase_latin_letters_with_diacritics(
        path="../data/lowercase_latin_letters_with_diacritics/test",
        subset=None,
        validation_split=None,
    )

    class_labels = [get_class_name(cn) for cn in validation_data.class_names]

    if clear_project_before:
        logger.info(f"Cleaning wandb project {wandb_project}")
        api = wandb.Api()
        runs = api.runs(f"gratkadlafana/{wandb_project}")
        for run in runs:
            run.delete()
        logger.info("Project runs deleted")

    with wandb.init(project=wandb_project, config={"class_labels": class_labels}):
        log_dataset_statistics(train_data, validation_data, class_labels)

    model_names = []
    model_accuracies = []
    no_train = ["CNN_1", "CNN_3", "CNN_5_alternative_arch_1"]
    for model, config in get_models_for_experiment():
        if config["model_name"] in no_train:
            continue
        wandb.init(project=wandb_project, config=config, name=config["model_name"])
        num_epochs = config["num_epochs"]
        history = train(
            train_data,
            model,
            epochs=num_epochs,
            validation_dataset=validation_data,
            callbacks=[
                WandbCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=2, restore_best_weights=True
                ),
            ],
        )

        _, y_true, y_pred, y_probas = test(test_data, model)
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred)
        val_acc = history.history["val_categorical_accuracy"][-1]
        model_names.append(config["model_name"])
        model_accuracies.append(val_acc)
        wandb.finish()

    with wandb.init(project=wandb_project):
        # post experiment visualisations
        plot_classification_accuracy(model_names, model_accuracies)


if __name__ == "__main__":
    run()
