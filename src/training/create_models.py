from typing import List, Tuple
import tensorflow as tf


def get_convnet_model(
    cnn_filters: list,
    cnn_kernels: list,
    dense_units: list,
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    dropout_rate: float,
    model_name: str,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
) -> Tuple[tf.keras.Model, dict]:
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

    config = dict(
        learning_rate=learning_rate,
        optimizer=str(optimizer),
        loss=loss.name,
        architecture="CNN",
        entity=model_name,
    )

    return model, config


def get_models_for_experiment() -> List:
    """
    Create models with different hyperparameters to be trained from scratch
    """
    models = [
        get_convnet_model(
            cnn_filters=[32, 64],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[128],
            input_shape=(32, 32, 3),
            num_classes=10,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name=f"CNN_test",
        )
    ]
    return models
