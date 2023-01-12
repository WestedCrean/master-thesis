import tensorflow as tf
import numpy as np


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)
        self.main_layers = [
            tf.keras.layers.Conv2D(
                filters, strides=strides, kernel_size=3, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            self.activation,
            tf.keras.layers.Conv2D(
                filters, strides=1, kernel_size=3, padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
        ]
        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                tf.keras.layers.Conv2D(
                    filters,
                    strides=strides,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                ),
                tf.keras.layers.BatchNormalization(),
            ]

    def call(self, inputs):
        x = inputs
        for layer in self.main_layers:
            x = layer(x)
        skip_x = inputs
        for layer in self.skip_layers:
            skip_x = layer(skip_x)
        return self.activation(x + skip_x)


class ResNet(tf.keras.Model):
    def __init__(
        self,
        block_design=[3, 4, 6, 3],
        input_shape=[32, 32, 1],
        num_classes=88,
        **kwargs
    ):
        super(ResNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.classifier = tf.keras.layers.Dense(num_classes, activation="softmax")
        self.input_conv = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    64,
                    kernel_size=7,
                    strides=2,
                    input_shape=input_shape,
                    padding="same",
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
            ]
        )
        self.resnet_blocks = tf.keras.models.Sequential()
        prev_filters = 64
        for filters in np.repeat(np.array([64, 128, 256, 512]), block_design):
            strides = 1 if filters == prev_filters else 2
            self.resnet_blocks.add(ResidualBlock(filters, strides=strides))
            prev_filters = filters
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs):
        x = self.input_conv(inputs)
        x = self.resnet_blocks(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def get_resnet_model(input_shape, block_design, num_classes) -> tf.keras.Sequential:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, kernel_size=7, strides=2, input_shape=input_shape, padding="same"
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding="same"),
        ]
    )
    resnet_blocks = tf.keras.models.Sequential()
    prev_filters = 64
    for filters in np.repeat(np.array([64, 128, 256, 512]), block_design):
        strides = 1 if filters == prev_filters else 2
        resnet_blocks.add(ResidualBlock(filters, strides=strides))
        prev_filters = filters
    model.add(resnet_blocks)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
    return model
