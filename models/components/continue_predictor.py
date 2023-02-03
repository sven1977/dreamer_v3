"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.components.mlp import MLP


class ContinuePredictor(tf.keras.Model):
    def __init__(self, *, model_dimension: Optional[str] = "XS"):
        super().__init__()
        self.mlp = MLP(model_dimension=model_dimension, output_layer_size=1)

    def call(self, h, z, return_distribution=False):
        """TODO

        Args:
            h: The deterministic hidden state of the sequence model. [B, dim(h)].
            z: The stochastic discrete representations of the original
                observation input. [B, num_categoricals, num_classes].
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z.shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        # Send h-cat-z through MLP.
        out = self.mlp(out)
        # Remove the extra [B, 1] dimension at the end to get a proper Bernoulli
        # distribution. Otherwise, tfp will think that the batch dims are [B, 1]
        # where they should be just [B].
        out= tf.squeeze(out, axis=-1)
        # Transform to a single prob value for a Bernoulli distribution.
        prob = tf.nn.sigmoid(out)
        bernoulli = tfp.distributions.Bernoulli(prob)
        # Draw a sample.
        continue_ = bernoulli.sample()

        # Return Bernoulli sample (whether to continue) OR (continue?, Bernoulli prob).
        if return_distribution:
            return continue_, bernoulli
        return continue_


if __name__ == "__main__":
    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))

    model = ContinuePredictor()

    out = model(h, z)
    print(out)

    out = model(h, z, return_bernoulli_prob=True)
    print(out)
