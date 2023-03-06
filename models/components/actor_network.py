"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from models.components.mlp import MLP


class ActorNetwork(tf.keras.Model):
    def __init__(
        self,
        *,
        action_space: gym.Space,
        model_dimension: Optional[str] = "XS",
        return_normalization_decay: float = 0.99,
    ):
        super().__init__()

        self.model_dimension = model_dimension
        # The EMA decay rate used for the [Percentile(R, 95%) - Percentile(R, 5%)]
        # diff to scale value targets for the actor loss.
        self.return_normalization_decay = return_normalization_decay
        self.ema_value_target_pct5 = tf.Variable(
            np.nan, dtype=tf.float32, trainable=False
        )
        self.ema_value_target_pct95 = tf.Variable(
            np.nan, dtype=tf.float32, trainable=False
        )

        self.action_space = action_space
        if isinstance(action_space, Discrete):
            output_layer_size = action_space.n
            self.mlp = MLP(
                model_dimension=self.model_dimension,
                output_layer_size=output_layer_size,
            )
        elif isinstance(action_space, Box):
            assert np.all(action_space.low) == 0.0 and np.all(action_space.high) == 1.0
            output_layer_size = np.prod(action_space.shape)
            self.mlp = MLP(
                model_dimension=self.model_dimension,
                output_layer_size=output_layer_size,
            )
            self.std_mlp = MLP(
                model_dimension=self.model_dimension,
                output_layer_size=output_layer_size,
            )
        else:
            raise ValueError(f"Invalid action space: {action_space}")

    def call(self, h, z, return_distribution=False):
        """TODO

        Args:
            h: The deterministic hidden state of the sequence model. [B, dim(h)].
            z: The stochastic discrete representations of the original
                observation input. [B, num_categoricals, num_classes].
        """
        # Flatten last two dims of z.
        assert len(z.shape) == 3
        z_shape = tf.shape(z)
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z_shape[0], -1))
        assert len(z.shape) == 2
        out = tf.concat([h, z], axis=-1)
        # Send h-cat-z through MLP.
        action_logits = self.mlp(out)

        if isinstance(self.action_space, Discrete):
            action_probs = tf.nn.softmax(action_logits)

            # Add the unimix weighting (1% uniform) to the probs.
            # See [1]: "Unimix categoricals: We parameterize the categorical distributions
            # for the world model representations and dynamics, as well as for the actor
            # network, as mixtures of 1% uniform and 99% neural network output to ensure
            # a minimal amount of probability mass on every class and thus keep log
            # probabilities and KL divergences well behaved."
            action_probs = 0.99 * action_probs + 0.01 * (1.0 / self.action_space.n)

            # Create the distribution object using the unimix'd probs.
            distr = tfp.distributions.OneHotCategorical(probs=action_probs)
            # Note: This distribution is reparametrized in the original implementation
            # tfp.distributions.Categorical is NOT reparametrized by default
            action = distr.sample()
            reparam = tf.cast(
                action_probs - tf.stop_gradient(action_probs), action.dtype
            )
            action = tf.stop_gradient(action) + reparam
            # Convert from onehot to integer
            action = tf.argmax(action, axis=-1)

        elif isinstance(self.action_space, Box):
            std_logits = self.std_mlp(out)
            minstd = 0.1
            maxstd = 1.0
            std_logits = (maxstd - minstd) * tf.sigmoid(std_logits + 2.0) + minstd
            distr = tfp.distributions.Normal(tf.tanh(action_logits), std_logits)
            distr = tfp.distributions.Independent(distr, len(self.action_space.shape))
            action = distr.sample()

        if return_distribution:
            return action, distr
        return action


if __name__ == "__main__":
    action_space = gym.spaces.Discrete(5)
    print("action space: ", action_space)

    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))

    model = ActorNetwork(action_space=action_space, model_dimension="XS")

    actions = model(h, z)
    print(actions)

    actions, distr = model(h, z, return_distribution=True)
    print(actions, distr.sample())
    print(distr.logits)

    action_space = gym.spaces.Box(0, 1, (5,))
    print("action space: ", action_space)

    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))

    model = ActorNetwork(action_space=action_space, model_dimension="XS")

    actions = model(h, z)
    print(actions)

    actions, distr = model(h, z, return_distribution=True)
    print(actions, distr.sample())
    print(distr.distribution.loc, distr.distribution.scale)
