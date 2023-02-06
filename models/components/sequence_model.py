"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
from typing import Optional

import gymnasium as gym
import numpy as np
import tensorflow as tf

from utils.model_sizes import get_gru_units


# TODO: de-hardcode the discrete action processing (currently one-hot).
class SequenceModel(tf.keras.Model):
    """The "sequence model" of the RSSM, computing ht+1 given (ht, zt, at).

    Here, h is the GRU unit output.
    """
    def __init__(
        self,
        *,
        model_dimension: Optional[str] = "XS",
        action_space: gym.Space,
        num_gru_units: Optional[int] = None,
    ):
        super().__init__()

        num_gru_units = get_gru_units(model_dimension, default=num_gru_units)

        self.action_space = action_space
        self.gru_unit = tf.keras.layers.GRU(
            num_gru_units,
            return_sequences=False,
            return_state=False,
            #activation=tf.nn.silu,
            #recurrent_activation=tf.nn.silu,
        )
        # Add layer norm after the GRU output.
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, z, a, h=None):
        """

        Args:
            z: The sequence of stochastic discrete representations of the original
                observation input. Note: `z` is not used for the dynamics predictor
                model (which predicts z from h).
            a: The sequence of previous action, discrete components will be
                one-hot encoded.
            h: The previous deterministic hidden state of the sequence model.
        """
        # Discrete int actions -> one_hot
        if isinstance(self.action_space, gym.spaces.Discrete):
            a = tf.one_hot(a, depth=self.action_space.n)
        # Flatten last two dims of z.
        assert len(z.shape) == 4
        z_shape = tf.shape(z)
        z = tf.reshape(tf.cast(z, tf.float32), shape=(z_shape[0], z_shape[1], -1))
        assert len(z.shape) == 3
        assert len(a.shape) == 3
        out = tf.concat([z, a], axis=-1)
        # Pass through GRU.
        out = self.gru_unit(out, initial_state=h)
        # Pass through LayerNorm.
        return self.layer_norm(out)


if __name__ == "__main__":
    # DreamerV2/3 Atari input space: B x 32 (num_categoricals) x 32 (num_classes)
    B = 1
    T = 3
    h_dim = 32
    num_categoricals = num_classes = 8
    h_tm1 = tf.convert_to_tensor(np.random.random(size=(B, 32)), dtype=tf.float32)
    z_seq = np.random.random(size=(B, T, num_categoricals, num_classes))
    a_space = gym.spaces.Discrete(4)
    a_seq = np.array([[a_space.sample() for t in range(T)] for b in range(B)])
    model = SequenceModel(action_space=a_space, num_gru_units=h_dim)
    h, last_h = model(z=z_seq, a=a_seq, h=h_tm1)
    print(h.shape)
