"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import gymnasium as gym
import tensorflow as tf
import tensorflow_probability as tfp

from models.components.cnn_atari import CNNAtari
from models.components.continue_predictor import ContinuePredictor
from models.components.conv_transpose_atari import ConvTransposeAtari
from models.components.dynamics_predictor import DynamicsPredictor
from models.components.representation_layer import RepresentationLayer
from models.components.reward_predictor import RewardPredictor
from models.components.sequence_model import SequenceModel
from utils.symlog import symlog


class AtariWorldModel(tf.keras.Model):
    def __init__(self, action_space: gym.Space, batch_length_T: int = 64):
        """TODO

        Args:
             action_space:
             batch_length_T:
        """
        super().__init__()

        self.batch_length_T = batch_length_T

        # RSSM (Recurrent State-Space Model)
        # Encoder + z-generator (x, h -> z).
        self.cnn_atari = CNNAtari()
        self.representation_layer = RepresentationLayer()
        # Dynamics predictor (h -> z^).
        self.dynamics_predictor = DynamicsPredictor()
        # Sequence Model (h-1, a-1, z-1 -> h).
        self.sequence_model = SequenceModel(action_space=action_space)

        # Reward Predictor.
        self.reward_predictor = RewardPredictor()
        # Continue Predictor.
        self.continue_predictor = ContinuePredictor()

        # Decoder (h, z -> x^).
        self.cnn_transpose_atari = ConvTransposeAtari()

    def call(self, inputs, *args, **kwargs):
        return self.forward_train(inputs, *args, **kwargs)

    def forward_inference(self, observations, actions, h):
        """TODO: inference

        All inputs have shape [B x ...] (no time dimension).

        Args:
            observations: The raw observations coming from the environment.
                These will be symlog'd prior to passing them through the encoder.
            actions: The actions that were taken given `observations`.
            h: The current deterministic internal state (h). This might be None
                if the episode just started (use initial zero-tensor).

        Returns:
            The next deterministic state (h(t+1)), as predicted by the sequence model.
        """
        # Compute bare encoder outs (not z; this done later with involvement of the
        # sequence model).
        # encoder_outs=[B, ...]
        encoder_out = self.cnn_atari(symlog(observations))
        repr_input = tf.concat([encoder_out, h], axis=-1)
        # Draw one z-sample (no need to return the distribution here).
        z = self.representation_layer(repr_input, return_z_probs=False)
        # Flatten z to [B, dim]:
        z = tf.reshape(z, shape=(z.shape[0], -1))
        assert len(z.shape) == 2
        # Compute next h using action and state.
        h_tp1 = self.sequence_model(z=z, a=actions, h=h)
        # Generate state from h and z.
        #state = tf.concat([h, z], axis=-1)
        # Compute predicted rewards and continue flags.
        #rewards = self.reward_predictor(h=h_tp1, z=?? <- z needs to be zt+1 here (from the encoder, NOT the dynamics model): "First, an encoder maps sensory inputs xt to stochastic representations zt. Then, a sequence model with recurrent state ht predicts the sequence of these representations given past actions at−1. The concate- nation of ht and zt forms the model state from which we predict rewards rt and episode continuation flags ct ∈ {0, 1} and reconstruct the inputs to ensure informative representations:")
        #continues = self.continue_predictor(h=h_tp1, z=z)
        # TODO: Probably should return predicted z^ (from dynamics model) here as well.
        #  As well as predicted rewards and continue flags (these are all part of the
        #  world model).
        return h_tp1

    def forward_train(self, observations, actions, initial_h, training=None):
        """TODO: training

        All inputs have shape (B x T x ...)
        """
        # Compute bare encoder outs (not z; this done later with involvement of the
        # sequence model).
        # encoder_outs=[B, T, ...]

        # Fold time dimension for CNN pass.
        B, T = observations.shape[0], observations.shape[1]
        observations = tf.reshape(observations, shape=[-1] + observations.shape.as_list()[2:])
        encoder_out = self.cnn_atari(symlog(observations))
        # Unfold time dimension.
        encoder_out = tf.reshape(encoder_out, shape=[-1, T] + encoder_out.shape.as_list()[1:])

        # Loop through the T-axis of our samples and perform one computation step at
        # a time. This is necessary because the sequence model's output (h(t+1)) depends
        # on the current z(t), but z(t) depends on the current sequence model's output
        # h(t).
        zs = []
        z_probs_encoder = []
        z_probs_dynamics = []
        hs = [initial_h or self._get_initial_h(batch_size=B)]
        h_tp1 = hs[-1]
        for t in range(self.batch_length_T):
            h_t = hs[-1]
            repr_input = tf.concat([encoder_out[:, t], h_t], axis=-1)
            # Draw one z-sample (z(t)) and also get the z-distribution for dynamics and
            # representation loss computations.
            z_t, z_probs = self.representation_layer(repr_input, return_z_probs=True)
            # z_t=[B, ]
            z_probs_encoder.append(z_probs)
            # Flatten z to [B, num_categoricals x num_classes]:
            zs.append(z_t)

            # Compute the predicted z_t (z^) using the dynamics model.
            _, z_probs = self.dynamics_predictor(h_t, return_z_probs=True)
            z_probs_dynamics.append(z_probs)

            # Compute h(t+1).
            # Make sure z- and action inputs to sequence model are sequences.
            # Expand to T=1:
            h_tp1 = self.sequence_model(
                z=tf.expand_dims(z_t, axis=1),
                a=actions[:, t:t+1],  # Make sure actions also has a T dimension.
                h=h_t,  # Initial state must NOT have a T dimension.
            )
            # Not needed for the last time step.
            if t < self.batch_length_T - 1:
                hs.append(h_tp1)

        # Stack at time dimension to yield: [B, T, ...].
        hs = tf.stack(hs, axis=1)
        zs = tf.stack(zs, axis=1)
        # Fold time axis to retrieve the final (loss ready) categorical distribution.
        z_probs_encoder = tf.stack(z_probs_encoder, axis=1)
        z_distribution_encoder = tfp.distributions.Categorical(
            logits=tf.reshape(z_probs_encoder, [-1] + z_probs_encoder.shape.as_list()[2:])
        )
        z_probs_dynamics = tf.stack(z_probs_dynamics, axis=1)
        z_distribution_dynamics = tfp.distributions.Categorical(
            logits=tf.reshape(z_probs_dynamics, [-1] + z_probs_dynamics.shape.as_list()[2:])
        )

        # Fold time dimension.
        hs = tf.reshape(hs, shape=[-1] + hs.shape.as_list()[2:])
        zs = tf.reshape(zs, shape=[-1] + zs.shape.as_list()[2:])

        # Compute predicted symlog'd observations from h and z using the decoder.
        _, obs_distribution = self.cnn_transpose_atari(
            h=hs, z=zs, return_distribution=True
        )
        # Compute (predicted) reward distributions.
        _, reward_distribution = self.reward_predictor(
            h=hs, z=zs, return_distribution=True
        )
        # Compute (predicted) continue distributions.
        _, continue_distribution = self.continue_predictor(
            h=hs, z=zs, return_distribution=True
        )

        # Return outputs for loss computation.
        # Note that all shapes are [B, ...] (no time axis).
        return {
            "obs_distribution": obs_distribution,
            "reward_distribution": reward_distribution,
            "continue_distribution": continue_distribution,
            "z_distribution_encoder": z_distribution_encoder,
            "z_distribution_dynamics": z_distribution_dynamics,
            # Next deterministic internal states (h). This needs to be passed into
            # the next call to this method.
            "h_tp1": h_tp1,
        }

    def _get_initial_h(self, batch_size: int):
        return tf.zeros(
            shape=(batch_size, self.sequence_model.gru_unit.units),
            dtype=tf.float32,
        )


if __name__ == "__main__":
    pass#world_model =
