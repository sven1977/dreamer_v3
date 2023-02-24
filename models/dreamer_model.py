"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import gymnasium as gym
import numpy as np
import tensorflow as tf

from models.components.actor_network import ActorNetwork
from models.components.critic_network import CriticNetwork
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

        self.model_dimension = model_dimension

        self.world_model = world_model

        self.action_space = action_space

        self.actor = ActorNetwork(
            action_space=self.action_space,
            model_dimension=self.model_dimension,
        )
        self.critic = CriticNetwork(
            model_dimension=self.model_dimension,
        )

    def call(self, inputs, *args, **kwargs):
        return self.forward_inference(inputs, *args, **kwargs)

    @tf.function
    def forward_inference(self, observations, initial_h, training=None):
        """TODO"""
        z_t = self.world_model.compute_posterior_z(observations, initial_h)

        # Compute action using our actor network.
        actions = self.actor(h=initial_h, z=z_t)

        # Compute next h using action and state.
        h_tp1 = self.world_model.sequence_model(
            # actions and z must have a T dimension.
            z=tf.expand_dims(z_t, axis=1),
            a=tf.expand_dims(actions, axis=1),
            h=initial_h,  # Initial state must NOT have a T dimension.
        )
        return actions, h_tp1

    @tf.function
    def forward_train(self, observations, actions, initial_h, training=None):
        """TODO"""
        return self.world_model.forward_train(observations, actions, initial_h)

    @tf.function
    def _get_initial_h(self, batch_size: int = 0):
        return self.world_model._get_initial_h(batch_size=batch_size)

    @tf.function
    def dream_trajectory(self, h, z, timesteps):
        """Dreams trajectories from batch of h- and z-states.

        Args:
            h: The h-states (B, ...) as computed by a train forward pass. From
                each individual h-state in the given batch, we will branch off
                a dreamed trajectory of len `timesteps`.
            z: The posterior z-states (B, num_categoricals, num_classes) as computed
                by a train forward pass. From each individual z-state in the given
                batch,, we will branch off a dreamed trajectory of len `timesteps`.
            timesteps: The number of timesteps to dream for.
        """
        assert h.shape[0] == z.shape[0], (
            "h- and z-shapes (batch size; 0th dim) must be the same!"
        )

        # Dreamed actions.
        a_dreamed_t1_to_H = []
        a_dreamed_distributions_t1_to_H = []
        # Dreamed rewards.
        r_dreamed_t1_to_Hp1 = []
        # Dreamed continue flags.
        c_dreamed_t1_to_Hp1 = []
        # Dreamed values.
        v_dreamed_t1_to_Hp1 = []
        # TODO: Make these just the probs. These distribution objects are not necessary.
        v_symlog_dreamed_distributions_t1_to_Hp1 = []
        v_symlog_dreamed_distributions_ema_t1_to_Hp1 = []

        # GRU outputs.
        h_states_t1_to_Hp1 = []
        # Dynamics model outputs.
        z_states_prior_t1_to_H = []

        for i in range(timesteps + 1):
            h_states_t1_to_Hp1.append(h)

            # Compute r using reward predictor.
            r = self.world_model.reward_predictor(h=h, z=z)
            r_dreamed_t1_to_Hp1.append(inverse_symlog(r))

            # Compute continues using continue predictor.
            c = self.world_model.continue_predictor(h=h, z=z)
            c_dreamed_t1_to_Hp1.append(c)

            # Compute the value estimates.
            v, v_distr = self.critic(h=h, z=z, return_distribution=True)
            v_dreamed_t1_to_Hp1.append(inverse_symlog(v))
            v_symlog_dreamed_distributions_t1_to_Hp1.append(v_distr)

            _, v_ema_distr = self.critic(
                h=h, z=z, return_distribution=True, use_ema=True
            )
            v_symlog_dreamed_distributions_ema_t1_to_Hp1.append(v_ema_distr)

            # Only for V, r, and continue flags, we need the values at H+1.
            # V(H+1): Needed for critic learning bootstrapping.
            # r(H+1): Needed for critic learning (V(H) depends on r(H+1) and V(H+1)).
            # c(H+1): Same as r(H+1).
            if i == timesteps:
                break

            # Compute `a` using actor network.
            a, a_dist = self.actor(h=h, z=z, return_distribution=True)
            # TEST: Use random actions instead of actor-computed ones.
            # a = tf.random.uniform(tf.shape(h)[0:1], 0, self.action_space.n, tf.int64)
            # END TEST: random actions
            a_dreamed_t1_to_H.append(a)
            a_dreamed_distributions_t1_to_H.append(a_dist)

            # Compute next h using sequence model.
            h_tp1 = self.world_model.sequence_model(
                # actions and z must have a T dimension.
                z=tf.expand_dims(z, axis=1),
                a=tf.expand_dims(a, axis=1),
                h=h,  # Initial state must NOT have a T dimension.
            )
            # Store z(t).
            z_states_prior_t1_to_H.append(z)
            # Compute z(t+1) from h(t+1), using the dynamics model.
            z = self.world_model.dynamics_predictor(h=h_tp1)

            h = h_tp1


        # Stack along T (horizon=H) axis.
        ret = {
            # Stop-gradient everything, except for the critic and action outputs.
            "h_states_t1_to_Hp1": tf.stop_gradient(
                tf.stack(h_states_t1_to_Hp1, axis=1)
            ),
            "z_states_prior_t1_to_H": tf.stop_gradient(
                tf.stack(z_states_prior_t1_to_H, axis=1)
            ),
            "rewards_dreamed_t1_to_Hp1": tf.stop_gradient(
                tf.stack(r_dreamed_t1_to_Hp1, axis=1)
            ),
            "continues_dreamed_t1_to_Hp1": tf.stop_gradient(
                tf.stack(c_dreamed_t1_to_Hp1, axis=1)
            ),
            # Critic and action outputs are not grad-stopped for critic/actor learning.
            "actions_dreamed_t1_to_H": tf.stack(a_dreamed_t1_to_H, axis=1),
            "actions_dreamed_distributions_t1_to_H": a_dreamed_distributions_t1_to_H,
            "values_dreamed_t1_to_Hp1": tf.stack(v_dreamed_t1_to_Hp1, axis=1),
            "values_symlog_dreamed_distributions_t1_to_Hp1": (
                v_symlog_dreamed_distributions_t1_to_Hp1
            ),
            "v_symlog_dreamed_distributions_ema_t1_to_Hp1": (
                v_symlog_dreamed_distributions_ema_t1_to_Hp1
            ),
        }

        return ret

    @tf.function
    def dream_trajectory_with_burn_in(self, observations, actions, initial_h, timesteps, use_sampled_actions=False):
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
            timesteps: The number of timesteps to dream for.
            use_sampled_actions: Whether to use `actions` for the dreamed predictions
                (rather than the actor network). If True, make sure that your
                `actions` arg contains as many timesteps as `observations` plus
                `timesteps` (b/c actions needs to cover both the burn-in phase
                as well as the actual dreaming phase).
        """
        #if use_sampled_actions:
        #    assert actions.shape[1] == observations.shape[1] + timesteps, (
        #        "Action timesteps provided ({actions.shape[1]}) seem wrong! Need "
        #        f"exactly {observations.shape[1] + timesteps}."
        #    )
        #else:
        #    assert actions.shape[1] == observations.shape[1], (
        #        "Action timesteps provided ({actions.shape[1]}) seem wrong! Need "
        #        f"exactly {observations.shape[1]}."
        #    )

        # Produce initial N internal states (burn-in):
        h = initial_h
        for i in range(observations.shape[1]):
            h = self.world_model.forward_inference(
                observations=observations[:, i],
                actions=actions[:, i],
                initial_h=h,
            )

        h_states_t1_to_Tp1 = [h]
        z_states_prior_t1_to_T = []
        a_dreamed_t1_to_T = []
        r_dreamed_t1_to_T = []
        r_symlog_dreamed_t1_to_T = []
        c_dreamed_t1_to_T = []
        for j in range(timesteps):
            actions_index = observations.shape[1] + j

            # Compute z from h, using the dynamics model (we don't have an actual
            # observation at this timestep).
            z = self.world_model.dynamics_predictor(h=h)
            z_states_prior_t1_to_T.append(z)

            # Compute r using reward predictor.
            r = self.world_model.reward_predictor(h=h, z=z)
            r_symlog_dreamed_t1_to_T.append(r)
            r_dreamed_t1_to_T.append(inverse_symlog(r))

            # Compute continues using continue predictor.
            c = self.world_model.continue_predictor(h=h, z=z)
            c_dreamed_t1_to_T.append(c)

            # Use the actions given to us.
            if use_sampled_actions:
                a = actions[:, actions_index]
            # Compute `a` using actor network.
            else:
                #a = self.actor(h=h, z=z)
                #TODO: compute actor-produced actions, instead of random actions
                a = tf.random.uniform(tf.shape(r), 0, self.action_space.n, tf.int64)
                #TODO: END: random actions

            a_dreamed_t1_to_T.append(a)

            # Compute next h using sequence model.
            h_tp1 = self.world_model.sequence_model(
                # actions and z must have a T dimension.
                z=tf.expand_dims(z, axis=1),
                a=tf.expand_dims(a, axis=1),
                h=h,  # Initial state must NOT have a T dimension.
            )
            h_states_t1_to_Tp1.append(h_tp1)

            h = h_tp1

        # Stack along T axis.
        ret = {
            # Note that h-states has one more entry as it includes the next h-state
            # ("reaching into" the next chunk). This very last h-state can be used
            # to start a new (dreamed) trajectory.
            "h_states_t1_to_Tp1": tf.stack(h_states_t1_to_Tp1, axis=1),
            "z_states_prior_t1_to_T": tf.stack(z_states_prior_t1_to_T, axis=1),
            "actions_dreamed_t1_to_T": tf.stack(a_dreamed_t1_to_T, axis=1),
            "rewards_dreamed_t1_to_T": tf.stack(r_dreamed_t1_to_T, axis=1),
            "rewards_symlog_dreamed_t1_to_T": tf.stack(
                r_symlog_dreamed_t1_to_T, axis=1
            ),
            "continues_dreamed_t1_to_T": tf.stack(c_dreamed_t1_to_T, axis=1)
        }

        return ret


if __name__ == "__main__":

    from IPython.display import display, Image
    from moviepy.editor import ImageSequenceClip
    import time

    from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

    from utils.env_runner import EnvRunner

    B = 1
    T = 64
    burn_in_T = 5

    config = (
        AlgorithmConfig()
            .environment("ALE/MsPacman-v5", env_config={
            # [2]: "We follow the evaluation protocol of Machado et al. (2018) with 200M
            # environment steps, action repeat of 4, a time limit of 108,000 steps per
            # episode that correspond to 30 minutes of game play, no access to life
            # information, full action space, and sticky actions. Because the world model
            # integrates information over time, DreamerV2 does not use frame stacking.
            # The experiments use a single-task setup where a separate agent is trained
            # for each game. Moreover, each agent uses only a single environment instance.
            # already done by MaxAndSkip wrapper "frameskip": 4,  # "action repeat" (frameskip) == 4
            "repeat_action_probability": 0.25,  # "sticky actions"
            "full_action_space": True,  # "full action space"
        })
        .rollouts(num_envs_per_worker=16, rollout_fragment_length=burn_in_T + T)
    )
    # The vectorized gymnasium EnvRunner to collect samples of shape (B, T, ...).
    env_runner = EnvRunner(model=None, config=config, max_seq_len=None, continuous_episodes=True)

    # Our DreamerV3 world model.
    #from_checkpoint = 'C:\Dropbox\Projects\dreamer_v3\examples\checkpoints\mspacman_world_model_170'
    from_checkpoint = "/Users/sven/Dropbox/Projects/dreamer_v3/examples/checkpoints/mspacman_dreamer_model_60"
    dreamer_model = tf.keras.models.load_model(from_checkpoint)
    world_model = dreamer_model.world_model
    # TODO: ugly hack (resulting from the insane fact that you cannot know
    #  an env's spaces prior to actually constructing an instance of it) :(
    env_runner.model = dreamer_model

    #obs = np.random.randint(0, 256, size=(B, burn_in_T, 64, 64, 3), dtype=np.uint8)
    #actions = np.random.randint(0, 2, size=(B, burn_in_T), dtype=np.uint8)
    #initial_h = np.random.random(size=(B, 256)).astype(np.float32)

    sampled_obs, _, sampled_actions, _, _, _, sampled_h = env_runner.sample(random_actions=False)

    dreamed_trajectory = dreamer_model.dream_trajectory(
        sampled_obs[:, :burn_in_T],
        sampled_actions.astype(np.int64),  # use all sampled actions, not random or actor-computed ones
        sampled_h,
        timesteps=T,
        # Use same actions as in the sample such that we can 100% compare
        # predicted vs actual observations.
        use_sampled_actions=True,
    )
    print(dreamed_trajectory)

    # Compute observations using h and z and the decoder net.
    # Note that the last h-state is NOT used here as it's already part of
    # a new trajectory.
    _, dreamed_images_distr = world_model.cnn_transpose_atari(
        tf.reshape(dreamed_trajectory["h_states_t1_to_Hp1"][:,:-1], (B * T, -1)),
        tf.reshape(dreamed_trajectory["z_dreamed_t1_to_Hp1"], (B * T) + dreamed_trajectory["z_dreamed_t1_to_Hp1"].shape[2:]),
    )
    # Use mean() of the Gaussian, no sample!
    #
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

    # Stitch dreamed_obs and sampled_obs together for better comparison.
    images = np.concatenate([dreamed_images, sampled_obs[:, burn_in_T:]], axis=2)

    # Save sequence a gif.
    clip = ImageSequenceClip(list(images[0]), fps=2)
    clip.write_gif("test.gif", fps=2)
    Image("test.gif")
    time.sleep(10)
