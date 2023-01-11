import tensorflow as tf


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2D(
                    out_channels,
                    kernel_size=3,
                    strides=stride,
                    padding="same",
                    input_shape=in_channels,
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )
        self.conv2 = tf.keras.layers.Sequential(
            [
                tf.keras.layers.Conv2D(
                    out_channels,
                    kernel_size=3,
                    strides=1,
                    padding="same",
                    input_shape=out_channels,
                ),
                tf.keras.layers.BatchNormalization(),
            ]
        )
        self.downsample = downsample
        self.relu = tf.keras.layers.ReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def call(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
