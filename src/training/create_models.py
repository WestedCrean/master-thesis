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
    use_dropout_features: bool = True,
    use_dropout_classifier: bool = True,
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
        if use_dropout_features:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    # Flatten
    x = tf.keras.layers.Flatten()(x)
    # Dense layers
    for units in dense_units:
        x = tf.keras.layers.Dense(units, activation="relu")(x)
        if use_dropout_classifier:
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
        model_name=model_name,
    )

    return model, config


def get_vgg_model(
    input_shape: tuple,
    num_classes: int,
    learning_rate: float,
    model_name: str,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
) -> Tuple[tf.keras.Model, dict]:
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Convolutional layers
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=64, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=128, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=256, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.Conv2D(
        filters=512, kernel_size=(3, 3), padding="same", activation="relu"
    )(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)
    x = tf.keras.layers.Dense(units=4096, activation="relu")(x)

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
        model_name=model_name,
    )

    return model, config


def get_models_for_experiment(num_classes=10, input_shape=(32, 32, 3)) -> List:
    """
    Create models with different hyperparameters to be trained from scratch
    """
    models = [
        get_convnet_model(
            cnn_filters=[32, 64],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[128],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name=f"CNN_32c_64c_128d",
        ),
        get_convnet_model(
            cnn_filters=[32, 64],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[128],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            use_dropout_features=False,
            model_name=f"CNN_32c_64c_128d_no_dropout_features",
        ),
        get_convnet_model(
            cnn_filters=[32, 64],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[128],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            use_dropout_features=False,
            use_dropout_classifier=False,
            model_name=f"CNN_32c_64c_128d_no_dropout",
        ),
        get_convnet_model(
            cnn_filters=[32, 64, 128],
            cnn_kernels=[(3, 3), (3, 3), (3, 3)],
            dense_units=[256, 128],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.3,
            model_name=f"CNN_32c_64c_128c_256d_128d",
        ),
    ]
    return models


def get_models_for_cnn_behavior_experiment(
    num_classes=10, input_shape=(32, 32, 3)
) -> List:
    """
    Create models with different hyperparameters to be trained from scratch
    """
    models = [
        get_convnet_model(
            cnn_filters=[],
            cnn_kernels=[],
            dense_units=[10],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_1",
        ),
        get_convnet_model(
            cnn_filters=[32],
            cnn_kernels=[(3, 3)],
            dense_units=[],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_2",
        ),
        get_convnet_model(
            cnn_filters=[32],
            cnn_kernels=[(3, 3)],
            dense_units=[10],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_3",
        ),
        get_convnet_model(
            cnn_filters=[64, 32],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[10],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_4",
        ),
        get_convnet_model(
            cnn_filters=[64, 32],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[64],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_5",
        ),
    ]
    return models
