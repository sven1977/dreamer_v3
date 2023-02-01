"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf

from models.components.mlp import MLP
from utils.two_hot import two_hot


class RewardPredictorLayer(tf.keras.layers.Layer):
    """A layer outputting reward predictions using K bins and two-hot encoding.
    TODO:
    """
    def __init__(
        self,
        num_buckets: int = 255,
        lower_bound: float = -20.0,
        upper_bound: float = 20.0,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.reward_buckets_layer = tf.keras.layers.Dense(
            self.num_buckets, activation=None
        )
        # Size of each reward bucket.
        self.bucket_delta = (
            (self.upper_bound - self.lower_bound) / (self.num_buckets - 1)
        )

    def call(self, inputs_, return_distribution=False):
        """Computes a predicted reward outputting weights for N discrete reward buckets.

        Args:
            inputs_: The input tensor for the layer, which computes the reward bucket
                weights (logits).
        """
        # Compute the `num_buckets` weights.
        out = self.reward_buckets_layer(inputs_)
        # Return the expected reward using softmax
        # out=[B, `num_buckets`]
        weights = tf.nn.softmax(out)
        # weights=[B, `num_buckets`]
        weighted_values = weights * tf.expand_dims(tf.range(self.lower_bound, self.upper_bound + self.bucket_delta, self.bucket_delta), 0)
        r = tf.reduce_sum(weighted_values, axis=-1)
        if return_distribution:
            return r, weighted_values
        return r


if __name__ == "__main__":
    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))
    inputs_ = tf.concat([h, tf.reshape(z, (1, 64))], -1)
    model = RewardPredictorLayer(num_buckets=5, lower_bound=-2.0, upper_bound=2.0)

    out = model(inputs_)
    print(out)

    out = model(inputs_, return_weighted_values=True)
    print(out)
