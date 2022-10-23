import yaml
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from engine import train, test
from src.datasets.phcd import numbers


def get_convnet_model(
    cnn_filters: list,
    cnn_kernels: list,
    dense_units: list,
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    dropout_rate: float,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
) -> tuple(tf.keras.Model, dict):
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # Convolutional layers
    for filters, kernel in zip(cnn_filters, cnn_kernels):
        x = tf.keras.layers.Conv2D(filters, kernel, activation="relu")(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Flatten
    x = tf.keras.layers.Flatten()(x)
    # Dense layers
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )

    config = dict(learning_rate=learning_rate, optimizer=optimizer._name, loss=loss.name, architecture="CNN")

    return model, config

def read_yaml_model_definitions() -> list:
    with open("./custom_models_definitions.yaml", "r") as f:
        model_definitions = yaml.safe_load(f)
    return [(model_name, model_definitions[model_name]) for model_name in model_definitions.keys()]

def __main__():
    model_definitions = read_yaml_model_definitions()
    for model_name, model_definition in model_definitions:
        model, config = get_convnet_model(**model_definition)
        print(f"Model: {model_name}")
        print(f"Config: {config}")
        model.summary()

        # log the model to wandb
        wandb.init(project="phcd_numbers", config=config, name=model_name)

        # train the model on numbers dataset
        model = train(
            dataset=numbers.train,
            model=model,
            epochs=config.get("epochs", 10),
            callbacks=[WandbCallback()],
        )

        accuracy, confusion_matrix = test(
            dataset=numbers.test,
            model=model,
            loss_fn=tf.keras.losses.CategoricalCrossentropy(),
            class_names=numbers.class_names,
        )
        
        wandb.log({"accuracy": accuracy, "confusion_matrix": confusion_matrix})
        wandb.save(model_name)
        model.save(f"models/{model_name}")

        
        
if __name__ == "__main__":
    __main__()