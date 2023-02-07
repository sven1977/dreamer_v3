"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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
        """TODO:

        Args:
            num_buckets: The number of buckets to create. Note that the number of
                possible symlog'd outcomes from the used distribution is
                `num_buckets` + 1:
                lower_bound --bucket-- o[1] --bucket-- o[2] ... --bucket-- upper_bound
                o=outcomes
                lower_bound=o[0]
                upper_bound=o[num_buckets]
            lower_bound: The symlog'd lower bound for a possible reward value.
                Note that a value of -20.0 here already allows individual (actual env)
                rewards to be as low as -400M. Buckets will be created between
                `lower_bound` and `upper_bound`.
            upper_bound: The symlog'd upper bound for a possible reward value.
                Note that a value of +20.0 here already allows individual (actual env)
                rewards to be as high as 400M. Buckets will be created between
                `lower_bound` and `upper_bound`.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.reward_buckets_layer = tf.keras.layers.Dense(
            units=self.num_buckets + 1,  # num_outcomes=num_buckets+1
            activation=None,
            # From [1]:
            # "We further noticed that the randomly initialized reward predictor and
            # critic networks at the start of training can result in large predicted
            # rewards that can delay the onset of learning. We initialize the output
            # weights of the reward predictor and critic to zeros, which effectively
            # alleviates the problem and accelerates early learning."
            kernel_initializer="zeros",
            bias_initializer="zeros",  # default anyways
        )
        # Size of each reward bucket.
        self.bucket_delta = (
            (self.upper_bound - self.lower_bound) / self.num_buckets
        )

    def call(self, inputs_, return_distribution=False):
        """Computes a distribution over N equal sized buckets reward.

        TODO: which tfp distribution do we use here?

        Args:
            inputs_: The input tensor for the layer, which computes the reward bucket
                weights (logits). [B, dim].
            return_distribution: Whether to return the FiniteDiscrete reward
                distribution object as a second return value (besides a reward sample).
        """
        # Compute the `num_buckets` weights.
        assert len(inputs_.shape) == 2
        out = self.reward_buckets_layer(inputs_)
        # Return the expected reward using softmax
        # out=[B, `num_buckets`]
        probs = tf.nn.softmax(out)
        # weights=[B, `num_buckets`]
        distr = tfp.distributions.FiniteDiscrete(
            outcomes=tf.linspace(
                self.lower_bound,
                self.upper_bound,
                self.num_buckets + 1,
            ),
            probs=probs,
            # Make the tolerance exactly half of the bucket delta.
            # This way, we should be able to compute the log_prob of any arbitrary
            # continuous value, even if it's not exactly an `outcomes` value.
            atol=self.bucket_delta / 2.0,
        )
        r_sample = distr.sample()
        # Note: In order to get the actually expected value, just do this with the
        # returned distribution:
        # `distr.mean()` OR `tf.reduce_sum(distr.probs * distr.outcomes)`
        if return_distribution:
            return r_sample, distr
        return r_sample


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
