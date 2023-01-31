"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
import tensorflow as tf


class MLP(tf.keras.Model):
    def __init__(self, output_layer_size=None):
        super().__init__()

        self.dense_layers = []
        for _ in range(1):
            self.dense_layers.append(tf.keras.layers.Dense(256, activation=tf.nn.silu))
        self.layer_normalizations = []
        for _ in range(len(self.dense_layers)):
            self.layer_normalizations.append(tf.keras.layers.LayerNormalization())

        self.output_layer = None
        if output_layer_size:
            self.output_layer = tf.keras.layers.Dense(
                output_layer_size, activation=None,
            )

    def call(self, h, z=None):
        """

        Args:
            h: The deterministic hidden state of the sequence model.
            z: The stochastic discrete representation of the original observation
                input. Note: `z` is not used for the dynamics predictor model (which
                predicts z from h).
        """
        if z is not None:
            out = tf.concat([h, z], axis=-1)
        else:
            out = h

        for dense_layer, layer_norm in zip(self.dense_layers, self.layer_normalizations):
            out = layer_norm(dense_layer(out))

        if self.output_layer is not None:
            out = self.output_layer(out)

        return out
