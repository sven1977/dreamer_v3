"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.components.mlp import MLP


class ContinuePredictor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.mlp = MLP(output_layer_size=1)

    def call(self, h, z, return_bernoulli_prob=False):
        """TODO

        Args:
            h: The deterministic hidden state of the sequence model.
            z: The stochastic discrete representations of the original
                observation input.
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z.shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        # Send h-cat-z through MLP.
        out = self.mlp(out)
        prob = tf.nn.sigmoid(out)
        distribution = tfp.distributions.Bernoulli(prob)
        continue_ = distribution.sample()
        # Return Bernoulli sample (whether to continue) OR (continue?, Bernoulli prob).
        if return_bernoulli_prob:
            return continue_, prob
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
