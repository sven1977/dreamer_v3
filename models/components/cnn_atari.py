"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf


class CNNAtari(tf.keras.Model):
    # TODO: Un-hard-code all hyperparameters, such as input dims, activation,
    #  filters, etc..
    def __init__(self):
        super().__init__()
        # See appendix C in [1]:
        # "We use a similar network architecture but employ layer normalization and
        # SiLU as the activation function. For better framework support, we use
        # same-padded convolutions with stride 2 and kernel size 3 instead of
        # valid-padded convolutions with larger kernels ..."
        self.conv_layers = [
            tf.keras.layers.Conv2D(
                filters=24,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            tf.keras.layers.Conv2D(
                filters=48,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            tf.keras.layers.Conv2D(
                filters=72,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            # .. until output is 4 x 4 x [num_filters].
            tf.keras.layers.Conv2D(
                filters=96,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
        ]
        self.layer_normalizations = []
        for _ in range(len(self.conv_layers)):
            self.layer_normalizations.append(tf.keras.layers.LayerNormalization())
        # -> 4 x 4 x num_filters -> now flatten.
        self.flatten_layer = tf.keras.layers.Flatten(data_format="channels_last")

    def call(self, inputs):
        out = inputs
        for conv_2d, layer_norm in zip(self.conv_layers, self.layer_normalizations):
            out = layer_norm(inputs=conv_2d(out))
        return self.flatten_layer(out)


if __name__ == "__main__":
    # World Models (and DreamerV2/3) Atari input space: 64 x 64 x 3
    inputs = np.random.random(size=(1, 64, 64, 3))
    model = CNNAtari()
    out = model(inputs)
    print(out.shape)
