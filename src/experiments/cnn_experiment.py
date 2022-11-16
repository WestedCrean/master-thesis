from pathlib import Path

import wandb
from wandb.keras import WandbCallback

from training.engine import train, test
from training.create_models import get_models_for_cnn_behavior_experiment
from datasets import (
    numbers,
    get_class_name,
    log_dataset_statistics,
)
from visualisations.history import plot_history
from visualisations.classification_metrics import get_classification_report


def run():
    wandb_project = "cnn_experiment"
    train_data = numbers(subset="training")
    validation_data = numbers(subset="validation")
    test_data = numbers(path="../data/numbers/test", subset=None, validation_split=None)

    class_labels = [get_class_name(cn) for cn in validation_data.class_names]

    with wandb.init(project=wandb_project, config={"class_labels": class_labels}):
        log_dataset_statistics(train_data, validation_data, class_labels)

    for model, config in get_models_for_cnn_behavior_experiment():
        wandb.init(project=wandb_project, config=config, name=config["model_name"])
        history = train(
            train_data,
            model,
            epochs=10,
            validation_dataset=validation_data,
            callbacks=[WandbCallback()],
        )

        _, y_true, y_pred, y_probas = test(test_data, model)
        wandb.sklearn.plot_confusion_matrix(y_true, y_pred)
        wandb.sklearn.plot_roc(y_true, y_probas, class_labels)
        cl = get_classification_report(y_true, y_pred, class_labels=class_labels)
        wandb.log(
            {
                "classification_report": cl,
                "precision_history": wandb.plot.line(
                    cl, "epoch", "precision", title="Precision History"
                ),
                "recall_history": wandb.plot.line(
                    cl, "epoch", "recall", title="Recall History"
                ),
                "f1_score_history": wandb.plot.line(
                    cl, "epoch", "f1-score", title="F1 Score History"
                ),
                "accuracy": plot_history(history),
            }
        )
        wandb.log(
            {
                "final_training_accuracy": history.history["accuracy"][-1],
                "final_validation_accuracy": history.history["val_accuracy"][-1],
            }
        )
        wandb.finish()


if __name__ == "__main__":
    run()
