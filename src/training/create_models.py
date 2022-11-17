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
    cnn_padding: str = "valid",
    cnn_activation_fn: str = "relu",
    pool_size: tuple = (2, 2),
    batch_norm_momentum: float = 0.99,
    use_dropout_features: bool = True,
    use_dropout_classifier: bool = True,
    use_double_conv_layers: bool = False,
    use_batch_normalization: bool = False,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
) -> Tuple[tf.keras.Model, dict]:
    # Input layer
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    # Convolutional layers
    for filters, kernel in zip(cnn_filters, cnn_kernels):
        x = tf.keras.layers.Conv2D(
            filters, kernel, padding=cnn_padding, activation=cnn_activation_fn
        )(x)
        if use_double_conv_layers:
            x = tf.keras.layers.Conv2D(
                filters, kernel, padding=cnn_padding, activation=cnn_activation_fn
            )(x)
        if use_batch_normalization:
            x = tf.keras.layers.BatchNormalization(momentum=batch_norm_momentum)(x)
        x = tf.keras.layers.MaxPool2D(pool_size=pool_size)(x)
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


def get_convnet_alternative_architecture_1_model(
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

    for f in [32, 64, 128]:
        # Convolutional layers
        C1_1 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=1,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(x)
        C1_2 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=1,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(C1_1)
        C1_3 = tf.keras.layers.BatchNormalization()(C1_2)

        C2_1 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=3,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(x)
        C2_2 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=3,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(C2_1)
        C2_3 = tf.keras.layers.BatchNormalization()(C2_2)

        C3_1 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=5,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(x)
        C3_2 = tf.keras.layers.Conv2D(
            filters=f,
            kernel_size=5,
            kernel_initializer="he_uniform",
            padding="same",
            activation="relu",
        )(C3_1)
        C3_3 = tf.keras.layers.BatchNormalization()(C3_2)

        x = tf.keras.layers.Average()([C1_3, C2_3, C3_3])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

    # Flatten
    x = tf.keras.layers.Flatten()(x)

    # Dense layers
    x = tf.keras.layers.Dense(units=128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

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
            cnn_filters=[32, 64, 128],
            cnn_kernels=[(3, 3), (3, 3), (3, 3)],
            dense_units=[256, 128],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.3,
            model_name=f"CNN_1",
        ),
        get_convnet_model(
            cnn_filters=[],
            cnn_kernels=[],
            dense_units=[16, 32, 16],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="Dense_1",
        ),
        get_convnet_model(
            cnn_filters=[],
            cnn_kernels=[],
            dense_units=[64, 128, 256],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="Dense_2",
        ),
        get_convnet_model(
            cnn_filters=[64, 32],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[64],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_2",
        ),
        get_convnet_model(
            cnn_filters=[32, 64, 32],
            cnn_kernels=[(5, 5), (5, 5), (5, 5)],
            dense_units=[256],
            cnn_padding="same",
            use_double_conv_layers=True,
            batch_norm_momentum=0.15,
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.25,
            model_name="CNN_3",
        ),
        get_convnet_model(
            cnn_filters=[64, 64, 128, 128],
            cnn_kernels=[
                (3, 3),
                (3, 3),
                (3, 3),
                (3, 3),
            ],
            dense_units=[1152, 256],
            cnn_padding="same",
            cnn_activation_fn=tf.keras.layers.LeakyReLU(alpha=0.01),
            use_double_conv_layers=True,
            batch_norm_momentum=0.15,
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.25,
            model_name="CNN_4",
        ),
        get_convnet_alternative_architecture_1_model(
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            model_name="CNN_5_alternative_arch_1",
        ),
        get_convnet_model(
            cnn_filters=[64, 128, 256],
            cnn_kernels=[
                (3, 3),
                (3, 3),
                (3, 3),
            ],
            dense_units=[256, 128, 64],
            cnn_padding="same",
            cnn_activation_fn=tf.keras.layers.LeakyReLU(alpha=0.1),
            use_double_conv_layers=True,
            batch_norm_momentum=0.9,
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.25,
            model_name="CNN_6",
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
            dense_units=[16, 32, 16],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="Dense_1",
        ),
        get_convnet_model(
            cnn_filters=[64, 32],
            cnn_kernels=[(3, 3), (3, 3)],
            dense_units=[64],
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.2,
            model_name="CNN_1",
        ),
        get_convnet_model(
            cnn_filters=[32, 64, 32],
            cnn_kernels=[(5, 5), (5, 5), (5, 5)],
            dense_units=[256],
            cnn_padding="same",
            use_double_conv_layers=True,
            batch_norm_momentum=0.15,
            input_shape=input_shape,
            num_classes=num_classes,
            learning_rate=0.001,
            dropout_rate=0.25,
            model_name="CNN_2",
        ),
    ]
    return models
