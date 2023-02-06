"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import gymnasium as gym
import tensorflow as tf

from models.components.actor_network import ActorNetwork
from models.components.reward_predictor import RewardPredictor


class DreamerModel(tf.keras.Model):
    """TODO
    """
    def __init__(
            self,
            *,
            model_dimension: str = "XS",
            action_space: gym.Space,
            batch_length_T: int = 64,
            world_model,
    ):
        """TODO

        Args:
             model_dimension: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the different network sizes.
             action_space: The action space the our environment used.
             batch_length_T: The length (T) of the sequences used for training. The
                actual shape of the input data (e.g. rewards) is then: [B, T, ...],
                where B is the "batch size", T is the "batch length" (this arg) and
                "..." is the dimension of the data (e.g. (64, 64, 3) for Atari image
                observations). Note that a single sequence (within a batch) only ever
                contains continuous time-step data from one episode. Should an
                episode have ended inside a sequence, the reset of that sequence will be
                filled with zero-data.
        """
        super().__init__()

        assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
        self.model_dimension = model_dimension

        self.batch_length_T = batch_length_T

        self.world_model = world_model

        self.actor = ActorNetwork(
            action_space=action_space,
            model_dimension=model_dimension,
        )
        self.critic = RewardPredictor(
            model_dimension=model_dimension,
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward_train(inputs, *args, **kwargs)

    @tf.function
    def forward_inference(self, observations, actions, initial_h):
        """TODO"""
        pass

    @tf.function
    def forward_train(self, observations, actions, initial_h, training=None):
        """TODO"""
        pass

    #@tf.function
    def dream_trajectory(self, observations, actions, initial_h, timesteps):
        """Dreams trajectory from N initial observations and an initial h-state.

        Uses all T observations and actions (B, T, ...) to get a most accurate
        h-state to begin dreaming with (burn-in). After all N observations/actions are
        "used up", starts producing z-states using the dynamics model and predicted
        actions (instead of the encoder and the provided actions).

        Args:
            observations: The batch (B, T, ...) of observations to be passed through
                the encoder network to yield the inputs to the
                representation layer (which then can compute the first N z-states).
            actions: The batch (B, T, ...) of actions to be passed through the sequence
                model to yield the first N h-states.
            initial_h: The initial h-states (B, ...) (no time dimension!) h(t) to be
                used in combination with the first observation (o(t)) to yield
                z(t) and then - in combination with the first action (a(t)) and z(t)
                to yield the next h-state (h(t+1)) via the RSSM.
        """
        # Produce initial N internal states (burn-in):
        h = initial_h
        for i in range(observations.shape[1]):
            h = self.world_model.forward_inference(
                observations=observations[:, i],
                actions=actions[:, i],
                initial_h=h,
            )

        h_states = []
        z_dreamed = []
        a_dreamed = []
        r_dreamed = []
        for _ in range(timesteps):
            # Compute z using the dynamics model.
            z = self.world_model.dynamics_predictor(h=h)
            z_dreamed.append(z)
            # Compute a using actor network.
            a = self.actor(h=h, z=z)
            a_dreamed.append(a)
            # Compute r using reward predictor.
            r_dreamed.append(self.world_model.reward_predictor(h=h, z=z))
            # Compute next h using sequence model.
            h_tp1 = self.world_model.sequence_model(
                # actions and z must have a T dimension.
                z=tf.expand_dims(z, axis=1),
                a=tf.expand_dims(a, axis=1),
                h=h,  # Initial state must NOT have a T dimension.
            )
            h_states.append(h_tp1)

            h = h_tp1

        return {
            "h_states": tf.stack(h_states, axis=1),
            "z_dreamed": tf.stack(z_dreamed, axis=1),
            "actions_dreamed": tf.stack(a_dreamed, axis=1),
            "rewards_dreamed": tf.stack(r_dreamed, axis=1),
        }


if __name__ == "__main__":
    pass#world_model =
