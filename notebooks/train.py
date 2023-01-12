import wandb
import utils
from pathlib import Path


def evaluate_model(model, ds_test, model_name):
    """
    Evaluate model test loss, accuracy and other characteristics then log to wandb
    """
    flops = wandb.run.summary["GFLOPs"]
    disk_size = utils.calculate_model_size_on_disk(f"./artifacts/{model_name}.h5")
    num_parameters = utils.calculate_model_num_parameters(model)

    # evaluate model on ds_test and log to wandb
    test_loss, test_acc = model.evaluate(ds_test)

    wandb.log(
        {
            "test loss": test_loss,
            "test accuracy": test_acc,
            "number of parameters": num_parameters,
            "disk size": disk_size,
            "model flops": flops,
        }
    )


def evaluate_diacritics_performance(model, ds_test):
    """
    Evaluate model test loss, accuracy on letters with diacritics then log to wandb
    """
    diacritics = {
        62: "ą",
        63: "ć",
        64: "ę",
        65: "ł",
        66: "ń",
        67: "ó",
        68: "ś",
        69: "ź",
        70: "ż",
        71: "Ą",
        72: "Ć",
        73: "Ę",
        74: "Ł",
        75: "Ń",
        76: "Ó",
        77: "Ś",
        78: "Ź",
        79: "Ż",
    }

    # log test accuracy on these classes separately to wandb

    diacritics_acc = {}
    for diacritic_label in diacritics.keys():
        ds_test_diacritic = ds_test.filter(lambda x, y: y == diacritic_label)
        test_loss, test_acc = model.evaluate(ds_test_diacritic)
        diacritics_acc[diacritic_label] = {
            "loss": test_loss,
            "accuracy": test_acc,
            "label": diacritics[diacritic_label],
        }

    wandb.log(diacritics_acc)
