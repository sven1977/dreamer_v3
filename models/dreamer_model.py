"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import gymnasium as gym
import numpy as np
import tensorflow as tf

from models.components.actor_network import ActorNetwork
from models.components.reward_predictor import RewardPredictor
from utils.symlog import inverse_symlog


class DreamerModel(tf.keras.Model):
    """TODO
    """
    def __init__(
            self,
            *,
            model_dimension: str = "XS",
            action_space: gym.Space,
            world_model,
    ):
        """TODO

        Args:
             model_dimension: The "Model Size" used according to [1] Appendinx B.
                Use None for manually setting the different network sizes.
             action_space: The action space the our environment used.
        """
        super().__init__()

        assert model_dimension in [None, "XS", "S", "M", "L", "XL"]
        self.model_dimension = model_dimension

        self.world_model = world_model

        self.action_space = action_space

        self.actor = ActorNetwork(
            action_space=self.action_space,
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

        h_states = [h]
        z_dreamed = []
        a_dreamed = []
        r_dreamed = []
        for _ in range(timesteps):
            # Compute z from h, using the dynamics model (we don't have an actual
            # observation at this timestep).
            z = self.world_model.dynamics_predictor(h=h)
            z_dreamed.append(z)

            # Compute r using reward predictor.
            r = self.world_model.reward_predictor(h=h, z=z)
            r_dreamed.append(inverse_symlog(r))

            # Compute `a` using actor network.
            #a = self.actor(h=h, z=z)
            #TODO: compute actor-produced actions, instead of random actions
            a = tf.random.uniform(tf.shape(r), 0, self.action_space.n, tf.int64)
            #TODO: END: random actions

            a_dreamed.append(a)

            # Compute next h using sequence model.
            h_tp1 = self.world_model.sequence_model(
                # actions and z must have a T dimension.
                z=tf.expand_dims(z, axis=1),
                a=tf.expand_dims(a, axis=1),
                h=h,  # Initial state must NOT have a T dimension.
            )
            h_states.append(h_tp1)

            h = h_tp1

        # Stack along T axis.
        ret = {
            "h_states": tf.stack(h_states, axis=1),
            "z_dreamed": tf.stack(z_dreamed, axis=1),
            "actions_dreamed": tf.stack(a_dreamed, axis=1),
            "rewards_dreamed": tf.stack(r_dreamed, axis=1),
        }

        return ret


if __name__ == "__main__":
    from IPython.display import display, Image
    from moviepy.editor import ImageSequenceClip
    import time

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

    from models.world_model_atari import WorldModelAtari
    from utils.env_runner import EnvRunner

    B = 1
    T = 16
    burn_in_T = 4

    config = (
        AlgorithmConfig()
            .environment("ALE/MontezumaRevenge-v5", env_config={
            # DreamerV3 paper does not specify, whether Atari100k is run
            # w/ or w/o sticky actions, just that frameskip=4.
            "frameskip": 4,
            "repeat_action_probability": 0.0,
        })
        .rollouts(num_envs_per_worker=1, rollout_fragment_length=burn_in_T + T)
    )
    # The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
    env_runner = EnvRunner(model=None, config=config, max_seq_len=T)

    # Our DreamerV3 world model.
    from_checkpoint = "/Users/sven/Dropbox/Projects/dreamer_v3/examples/checkpoints/montezuma_world_model_330"
    world_model = tf.keras.models.load_model(from_checkpoint)
    # TODO: ugly hack (resulting from the insane fact that you cannot know
    #  an env's spaces prior to actually constructing an instance of it) :(
    env_runner.model = world_model

    dreamer_model = DreamerModel(
        model_dimension="S",
        action_space=env_runner.env.single_action_space,
        world_model=world_model,
    )
    #obs = np.random.randint(0, 256, size=(B, burn_in_T, 64, 64, 3), dtype=np.uint8)
    #actions = np.random.randint(0, 2, size=(B, burn_in_T), dtype=np.uint8)
    #initial_h = np.random.random(size=(B, 256)).astype(np.float32)

    sampled_obs, _, sampled_actions, _, _, _, sampled_h, _ = env_runner.sample(random_actions=True)

    dreamed_trajectory = dreamer_model.dream_trajectory(
        sampled_obs[:, :burn_in_T], sampled_actions.astype(np.int64)[:, :burn_in_T], sampled_h, timesteps=T
    )
    print(dreamed_trajectory)

    # Compute observations using h and z and the decoder net.
    # Note that the last h-state is NOT used here as it's already part of
    # a new trajectory.
    _, dreamed_images_distr = world_model.cnn_transpose_atari(
        tf.reshape(dreamed_trajectory["h_states"][:,:-1], (B * T, -1)),
        tf.reshape(dreamed_trajectory["z_dreamed"], (B * T) + dreamed_trajectory["z_dreamed"].shape[2:]),
    )
    # Use mean() of the Gaussian, no sample!
    dreamed_images = dreamed_images_distr.mean()
    dreamed_images = tf.reshape(
        tf.cast(
            tf.clip_by_value(
                inverse_symlog(dreamed_images), 0.0, 255.0
            ),
            tf.uint8,
        ),
        shape=(B, T, 64, 64, 3),
    ).numpy()

    # Save sequence a gif.
    clip = ImageSequenceClip(list(dreamed_images[0]), fps=20)
    clip.write_gif("test.gif", fps=20)
    Image("test.gif")
    time.sleep(10)
