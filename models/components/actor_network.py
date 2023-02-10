"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional

import gymnasium as gym
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
    ):
        super().__init__()

        self.model_dimension = model_dimension

        # TODO: For now, limit to discrete actions.
        assert isinstance(action_space, gym.spaces.Discrete)
        self.mlp = MLP(
            model_dimension=self.model_dimension,
            output_layer_size=action_space.n,
        )

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

        distr = tfp.distributions.Categorical(logits=action_logits)

        action = distr.sample()

        if return_distribution:
            return action, distr
        return action


if __name__ == "__main__":
    action_space = gym.spaces.Discrete(5)

    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))

    model = ActorNetwork(action_space=action_space, model_dimension="XS")

    actions = model(h, z)
    print(actions)

    actions, distr = model(h, z, return_distribution=True)
    print(actions, distr.sample(), distr.logits)
