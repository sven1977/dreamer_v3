"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class ConvTransposeAtari(tf.keras.Model):
    # TODO: Un-hard-code all hyperparameters, such as input dims, activation,
    #  filters, etc..
    def __init__(self):
        super().__init__()
        # See appendix B in [1]:
        # "The decoder starts with a dense layer, followed by reshaping
        # to 4 × 4 × C and then inverts the encoder architecture. ..."
        self.dense_layer = tf.keras.layers.Dense(4 * 4 * 192, activation=tf.nn.silu)
        # Inverse conv2d stack. See cnn_atari.py for Conv2D stack.
        self.conv_transpose_layers = [
            tf.keras.layers.Conv2DTranspose(
                filters=96,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=48,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            tf.keras.layers.Conv2DTranspose(
                filters=24,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
            # .. until output is 64 x 64 x 3.
            tf.keras.layers.Conv2DTranspose(
                filters=3,
                kernel_size=3,
                strides=(2, 2),
                padding="same",
                activation=tf.nn.silu,
            ),
        ]
        self.layer_normalizations = []
        for _ in range(len(self.conv_transpose_layers)):
            self.layer_normalizations.append(tf.keras.layers.LayerNormalization())

    def call(self, h, z):
        """TODO

        Args:
            h: The deterministic hidden state of the sequence model.
            z: The sequence of stochastic discrete representations of the original
                observation input. Note: `z` is not used for the dynamics predictor
                model (which predicts z from h).
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z.shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        # Feed through initial dense layer to get the right number of input nodes
        # for the first conv2dtranspose layer.
        out = self.dense_layer(out)
        # Reshape to image format.
        out = tf.reshape(out, shape=(-1, 4, 4, 192))
        # Pass through stack of Conv2DTransport layers (and layer norms).
        for conv_transpose_2d, layer_norm in zip(self.conv_transpose_layers, self.layer_normalizations):
            out = layer_norm(inputs=conv_transpose_2d(out))
        # Interpret output as means of a diag-Gaussian with std=1.0:
        # From [2]:
        # "Distributions The image predictor outputs the mean of a diagonal Gaussian
        # likelihood with unit variance, ..."
        distribution = tfp.distributions.Normal(loc=out, scale=1.0)
        return distribution.sample()


if __name__ == "__main__":
    # DreamerV2/3 Atari input space: B x 32 (num_categoricals) x 32 (num_classes)
    inputs = np.random.random(size=(1, 32, 32))
    model = ConvTransposeAtari()
    out = model(inputs)
    print(out.shape)
