from typing import List, Tuple
import tensorflow as tf
import tensorflow_hub as hub


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
    num_epochs: int = 8,
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
        num_epochs=num_epochs,
    )

    return model, config


def get_mobilenet_v2(
    input_shape: tuple,
    num_classes: int,
    model_name: str = "mobilenet_v2",
    num_epochs: int = 8,
    learning_rate: float = 0.001,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
):
    """
    Get MobileNetV2 feature extractor model from TensorFlow Hub.
    """
    mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

    feature_extractor_model = mobilenet_v2

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Preprocessing for mobilenet_v2
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)

    # Feature extractor layer
    x = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False
    )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
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
        num_epochs=num_epochs,
    )

    return model, config


def get_inception_v3(
    input_shape: tuple,
    num_classes: int,
    model_name: str = "inception_v3",
    num_epochs: int = 8,
    learning_rate: float = 0.001,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
):
    """
    Get InceptionV3 feature extractor model from TensorFlow Hub.
    """
    inception_v3 = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4"

    feature_extractor_model = inception_v3

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Preprocessing for inception_v3
    x = tf.keras.layers.experimental.preprocessing.Resizing(299, 299)(x)

    # Feature extractor layer
    x = hub.KerasLayer(
        feature_extractor_model, input_shape=(299, 299, 3), trainable=False
    )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
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
        num_epochs=num_epochs,
    )

    return model, config


def get_resnet50(
    input_shape: tuple,
    num_classes: int,
    model_name: str = "resnet50",
    num_epochs: int = 8,
    learning_rate: float = 0.001,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
):
    """
    Get ResNet50 feature extractor model from TensorFlow Hub.
    """
    resnet50 = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/5"

    feature_extractor_model = resnet50

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Preprocessing for resnet50
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)

    # Feature extractor layer
    x = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False
    )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
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
        num_epochs=num_epochs,
    )

    return model, config


def get_vgg16(
    input_shape: tuple,
    num_classes: int,
    model_name: str = "vgg16",
    num_epochs: int = 8,
    learning_rate: float = 0.001,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
):
    """
    Get VGG16 feature extractor model from TensorFlow Hub.
    """
    vgg16 = "https://tfhub.dev/emilutz/vgg19-block1-conv2-unpooling-encoder/1"

    feature_extractor_model = vgg16

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Feature extractor layer
    x = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False
    )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
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
        num_epochs=num_epochs,
    )

    return model, config


def get_efficientnet_v2(
    input_shape: tuple,
    num_classes: int,
    model_name: str = "efficientnet_v2",
    num_epochs: int = 8,
    learning_rate: float = 0.001,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam,
    loss: tf.keras.losses.Loss = tf.keras.losses.CategoricalCrossentropy(),
    metrics: list = [tf.keras.metrics.CategoricalAccuracy()],
):
    """
    Get EfficientNetV2 feature extractor model from TensorFlow Hub.
    """
    efficientnet_v2 = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"

    feature_extractor_model = efficientnet_v2

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # Preprocessing for efficientnet_v2
    x = tf.keras.layers.experimental.preprocessing.Resizing(224, 224)(x)

    # Feature extractor layer
    x = hub.KerasLayer(
        feature_extractor_model, input_shape=(224, 224, 3), trainable=False
    )(x)

    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
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
        num_epochs=num_epochs,
    )

    return model, config


def get_models_for_experiment(
    num_classes=10, input_shape=(32, 32, 3)
) -> List[tf.keras.Model]:
    return [
        get_mobilenet_v2(input_shape=input_shape, num_classes=num_classes),
        get_inception_v3(input_shape=input_shape, num_classes=num_classes),
        get_resnet50(input_shape=input_shape, num_classes=num_classes),
        get_vgg16(input_shape=input_shape, num_classes=num_classes),
        get_efficientnet_v2(input_shape=input_shape, num_classes=num_classes),
    ]
