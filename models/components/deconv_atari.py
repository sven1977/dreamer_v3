"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
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

    def call(self, inputs_):
        # `inputs_` are z tensors of shape (B, num_categoricals, num_classes)
        # Reshape such that we can use `out` as input to our conv-transpose stack.
        out = tf.reshape(inputs_, shape=(inputs_.shape[0], -1))
        out = self.dense_layer(out)
        out = tf.reshape(out, shape=(-1, 4, 4, 192))
        for conv_transpose_2d, layer_norm in zip(self.conv_transpose_layers, self.layer_normalizations):
            out = layer_norm(inputs=conv_transpose_2d(out))
        # Interpret output as means of a diag-Gaussian with std=1.0:
        distribution = tfp.distributions.Normal(loc=out, scale=1.0)
        return distribution.sample()


if __name__ == "__main__":
    # DreamerV2/3 Atari input space: B x 32 (num_categoricals) x 32 (num_classes)
    inputs = np.random.random(size=(1, 32, 32))
    model = ConvTransposeAtari()
    out = model(inputs)
    print(out.shape)
