import wandb
import numpy as np
import pathlib
import shutil
from typing import List
import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2, efficientnet

from training.utils import (
    load_data,
    get_number_of_classes,
    preprocess_dataset,
    calculate_model_size_on_disk,
    calculate_model_num_parameters,
)

params_for_hyperparameter_opt = {
    "model_name": {"values": ["b0", "b1", "b2", "b3"]},
    "efficientnet_version": {"values": [1, 2]},
}


def get_defaults():
    return dict(
        batch_size=32 * 4,
        epochs=100,
        optimizer="sgd",
        learning_rate=0.01,
        momentum=0.9,
        efficientnet_name="b3",
        efficientnet_version=2,
    )


def get_sweep_params():
    return {
        "method": "random",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "batch_size": {"values": [32, 128, 256]},
            "epochs": {"value": 1},
            "learning_rate": {
                "values": [
                    1e-2,
                    1e-3,
                    # 1e-4,
                ]
            },
            "optimizer": {
                "values": [
                    "sgd",
                    "adagrad",
                ]
            },
            "efficientnet_version": [1, 2],
            "efficientnet_name": {"values": ["b0", "b1", "b2", "b3"]},
        },
    }


def get_efficientnet_model(
    input_shape, num_classes, model_name="b0", efficientnet_version=2
) -> tf.keras.Sequential:
    base_model = None
    effnet_module = None

    if efficientnet_version == 2:
        effnet_module = efficientnet_v2
    else:
        effnet_module = efficientnet

    if model_name == "b0":
        base_model = effnet_module.EfficientNetV2B0(
            include_top=False, input_shape=input_shape, weights=None, pooling="max"
        )
    elif model_name == "b1":
        base_model = effnet_module.EfficientNetV2B1(
            include_top=False, input_shape=input_shape, weights=None, pooling="max"
        )
    elif model_name == "b2":
        base_model = effnet_module.EfficientNetV2B2(
            include_top=False, input_shape=input_shape, weights=None, pooling="max"
        )
    elif model_name == "b3":
        base_model = effnet_module.EfficientNetV2B3(
            include_top=False, input_shape=input_shape, weights=None, pooling="max"
        )
    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.Dense(num_classes),
            tf.keras.layers.Activation(
                "softmax", dtype="float32", name="predictions"
            ),  # this is needed for mixed precision training
        ]
    )
    return model


def train(
    config=None,
    job_type="sweep",
):
    with wandb.init(
        project="master-thesis",
        job_type=job_type,
        config=config,
        settings=wandb.Settings(start_method="thread"),
    ) as run:
        MODEL_NAME = run.name
        # hyperparameters
        opt_name = wandb.config.optimizer
        lr = wandb.config.learning_rate
        bs = wandb.config.batch_size
        epochs = wandb.config.epochs
        model_name = wandb.config.model_name
        efficientnet_version = wandb.config.efficientnet_version

        ds_train, ds_test, ds_val = load_data(run)

        num_classes = get_number_of_classes(ds_val)
        ds_train = preprocess_dataset(ds_train, batch_size=bs)
        ds_val = preprocess_dataset(ds_val, batch_size=bs)
        ds_test = preprocess_dataset(ds_test, batch_size=bs, cache=False)

        model = get_efficientnet_model(
            input_shape=(32, 32, 1),
            num_classes=num_classes,
            model_name=model_name,
            efficientnet_version=efficientnet_version,
        )

        opt = tf.keras.optimizers.get(
            {
                "class_name": opt_name,
                "config": {
                    "learning_rate": lr,
                },
            }
        )

        model.compile(
            optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        # save the best model
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=f"./artifacts/{MODEL_NAME}.h5",
            save_weights_only=False,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        )

        wandb_callback = wandb.keras.WandbCallback(
            save_model=False,
            compute_flops=True,
        )
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5
        )

        train_callbacks = [wandb_callback, early_stop]

        if job_type != "sweep":
            train_callbacks.append(checkpoint_callback)

        model.fit(
            ds_train,
            epochs=epochs,
            validation_data=ds_val,
            callbacks=train_callbacks,
        )

        # calculate model size on disk, flops and number of parameters
        flops = wandb.run.summary["GFLOPS"]
        disk_size = calculate_model_size_on_disk(f"./artifacts/{MODEL_NAME}.h5")
        num_parameters = calculate_model_num_parameters(model)

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
