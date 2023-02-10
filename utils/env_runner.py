"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from functools import partial
from typing import Optional

import gymnasium as gym
import numpy as np
from supersuit.generic_wrappers import resize_v1, color_reduction_v0
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.wrappers.atari_wrappers import EpisodicLifeEnv, FireResetEnv, NoopResetEnv, MaxAndSkipEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch


class CountEnv(gym.ObservationWrapper):
    def reset(self, **kwargs):
        self.__counter = 0
        return super().reset(**kwargs)

    def observation(self, observation):
        # For gray-scaled observations.
        observation[0][0] = self.__counter
        # For 3-color observations
        #observation[0][0][0] = self.__counter
        self.__counter += 1
        return observation


class NormalizeImageObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(-1.0, 1.0, (64, 64, 3), dtype=np.float32)

    def observation(self, observation):
        # Normalize and center (from -1.0 to 1.0).
        return (observation / 128) - 1.0


class EnvRunner:
    """An environment runner to locally collect data from vectorized gym environments.
    """

    def __init__(
        self,
        model,
        config: AlgorithmConfig,
        max_seq_len: Optional[int] = None,
        continuous_episodes: bool = False,
        _debug_count_env=False
    ):
        """Initializes an EnvRunner instance.

        Args:
            config: The config to use to setup this EnvRunner.
        """
        self.model = model
        self.config = config
        self.max_seq_len = max_seq_len
        self.continuous_episodes = continuous_episodes

        # If we are using continuous episodes (no zero-masks for timesteps past
        # end of an episode, just continue with next one), max_seq_len should be None.
        # Each sample() call will play out exactly rollout_fragment_length timesteps
        # per sub-environment.
        if self.continuous_episodes:
            assert self.max_seq_len is None
            self.max_seq_len = self.config.rollout_fragment_length

        if self.config.env.startswith("ALE"):
            # [2]: "We down-scale the 84 × 84 grayscale images to 64 × 64 pixels so that
            # we can apply the convolutional architecture of DreamerV1."
            # ...
            # "We follow the evaluation protocol of Machado et al. (2018) with 200M
            # environment steps, action repeat of 4, a time limit of 108,000 steps per
            # episode that correspond to 30 minutes of game play, no access to life
            # information, full action space, and sticky actions. Because the world
            # model integrates information over time, DreamerV2 does not use frame
            # stacking."
            wrappers = [
                partial(gym.wrappers.TimeLimit, max_episode_steps=108000),
                color_reduction_v0,  # grayscale
                partial(resize_v1, x_size=64, y_size=64),  # resize to 64x64
                NoopResetEnv,
                MaxAndSkipEnv,
            ]
            if _debug_count_env:
                wrappers.append(CountEnv)
            self.env = gym.vector.make(
                "GymV26Environment-v0",
                env_id=self.config.env,
                wrappers=wrappers,
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
                make_kwargs=self.config.env_config,
            )
        else:
            self.env = gym.vector.make(
                self.config.env,
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
                #make_kwargs=self.config.env_config,
            )
        self.needs_initial_reset = True
        self.observations = [[] for _ in range(self.env.num_envs)]
        self.actions = [[] for _ in range(self.env.num_envs)]
        self.rewards = [[] for _ in range(self.env.num_envs)]
        self.terminateds = [[] for _ in range(self.env.num_envs)]
        self.truncateds = [[] for _ in range(self.env.num_envs)]
        self.next_h_states = None
        self.current_sequence_initial_h = None

    def sample(self, explore: bool = True, random_actions: bool = False):
        if self.config.batch_mode == "complete_episodes":
            raise NotImplementedError
        else:
            return self.sample_timesteps(
                num_timesteps=(
                    self.config.rollout_fragment_length
                    * self.config.num_envs_per_worker
                ),
                explore=explore,
                random_actions=random_actions,
                force_reset=False,
            )

    def sample_timesteps(
        self,
        num_timesteps: int,
        explore: bool = True,
        random_actions: bool = False,
        force_reset: bool = False,
    ) -> MultiAgentBatch:
        """Runs n timesteps on the environment(s) and returns experiences.

        Timesteps are counted in total (across all vectorized sub-environments). For
        example, if self.num_envs=2 and num_timesteps=10, each sub-environment
        will be sampled for 5 steps.

        Args:
            num_timesteps: The number of timesteps to sample from the environment(s).
            explore: Indicates whether to utilize exploration when picking actions.
            force_reset: Whether to reset the environment(s) before starting to sample.
                If False, will still reset the environment(s) if they were left in
                a terminated or truncated state during previous sample calls.

        Returns:
            A MultiAgentBatch holding the collected experiences.
        """
        return_obs = []
        return_actions = []
        return_rewards = []
        return_terminateds = []
        return_truncateds = []
        return_initial_h = []
        return_next_obs = []
        if not self.continuous_episodes:
            return_masks = []

        if force_reset or self.needs_initial_reset:
            obs, _ = self.env.reset()
            if self.model is not None:
                self.next_h_states = self.model._get_initial_h(
                    batch_size=self.config.num_envs_per_worker
                ).numpy()
            else:
                self.next_h_states = np.array([0.0] * self.config.num_envs_per_worker)
            self.current_sequence_initial_h = self.next_h_states.copy()
            self.needs_initial_reset = False
            for i, o in enumerate(self._split_by_env(obs)):
                self.observations[i].append(o)
        else:
            obs = np.stack([o[-1] for o in self.observations])

        ts = 0
        ts_sequences = [len(self.observations[i]) - 1 for i in range(self.env.num_envs)]

        while True:
            if random_actions:
                # TODO: hack; right now, our model (a world model) does not have an
                #  actor head yet. Still perform a forward pass to get the next h-states.
                actions = self.env.action_space.sample()
                #print(f"took action {actions}")
                if self.model is not None:
                    self.next_h_states = (
                        self.model.forward_inference(obs, actions, tf.convert_to_tensor(self.next_h_states))
                    ).numpy()
                else:
                    self.next_h_states = np.array(ts_sequences)
            else:
                action_logits = self.model(obs)
                # Sample.
                if explore:
                    actions = tf.random.categorical(action_logits, num_samples=1)
                # Greedy.
                else:
                    actions = np.argmax(action_logits, axis=-1)

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

            for i, (o, a, r, term, trunc) in enumerate(zip(
                    self._split_by_env(obs),
                    self._split_by_env(actions),
                    self._split_by_env(rewards),
                    self._split_by_env(terminateds),
                    self._split_by_env(truncateds),
            )):
                self.observations[i].append(o)
                self.actions[i].append(a)
                self.rewards[i].append(r)
                self.terminateds[i].append(term)
                self.truncateds[i].append(trunc)
            # Make sure we always have one more obs stored than rewards (and actions)
            # due to the reset and last-obs logic of an MDP.
            assert(len(self.observations[0]) == len(self.rewards[0]) + 1)

            ts += self.env.num_envs
            for i in range(self.env.num_envs):
                ts_sequences[i] += 1

                if self.continuous_episodes:
                    if ts_sequences[i] == self.max_seq_len:
                        ts_sequences[i] = 0
                        return_obs.append(
                            self._process_and_pad(self.observations[i][:-1]))
                        return_actions.append(self._process_and_pad(self.actions[i]))
                        return_rewards.append(self._process_and_pad(self.rewards[i]))
                        return_terminateds.append(
                            self._process_and_pad(self.terminateds[i]))
                        return_truncateds.append(
                            self._process_and_pad(self.truncateds[i]))
                        return_initial_h.append(
                            self.current_sequence_initial_h[i].copy())

                        # The last entry in self.observations[i] is already the reset
                        # obs of the new episode.
                        if terminateds[i] or truncateds[i]:  #
                            return_next_obs.append(infos["final_observation"][i])
                            # Reset h-states to all zeros b/c we are starting a new episode.
                            if self.model is not None:
                                self.next_h_states[i] = self.model._get_initial_h(
                                    batch_size=0).numpy()
                            else:
                                self.next_h_states[i] = 0.0
                        # Last entry in self.observations[i] is the next obs (continuing
                        # the ongoing episode).
                        else:
                            return_next_obs.append(self.observations[i][-1])

                        self.observations[i] = [self.observations[i][-1]]
                        self.actions[i] = []
                        self.rewards[i] = []
                        self.terminateds[i] = []
                        self.truncateds[i] = []
                        self.current_sequence_initial_h[i] = self.next_h_states[
                            i].copy()

                else:
                    if terminateds[i] or truncateds[i] or ts_sequences[i] == self.max_seq_len:
                        return_masks.append(ts_sequences[i])
                        ts_sequences[i] = 0

                        return_obs.append(self._process_and_pad(self.observations[i][:-1]))
                        return_actions.append(self._process_and_pad(self.actions[i]))
                        return_rewards.append(self._process_and_pad(self.rewards[i]))
                        return_terminateds.append(self._process_and_pad(self.terminateds[i]))
                        return_truncateds.append(self._process_and_pad(self.truncateds[i]))
                        return_initial_h.append(self.current_sequence_initial_h[i].copy())

                        # The last entry in self.observations[i] is already the reset obs
                        # of the new episode.
                        if terminateds[i] or truncateds[i]:#
                            return_next_obs.append(infos["final_observation"][i])
                            # Reset h-states to all zeros b/c we are starting a new episode.
                            if self.model is not None:
                                self.next_h_states[i] = self.model._get_initial_h(batch_size=0).numpy()
                            else:
                                self.next_h_states[i] = 0.0
                        # Last entry in self.observations[i] is the next obs (continuing
                        # the ongoing episode).
                        else:
                            return_next_obs.append(self.observations[i][-1])

                        self.observations[i] = [self.observations[i][-1]]
                        self.actions[i] = []
                        self.rewards[i] = []
                        self.terminateds[i] = []
                        self.truncateds[i] = []
                        self.current_sequence_initial_h[i] = self.next_h_states[i].copy()

            # Make sure we always have one more obs stored than rewards (and actions)
            # due to the reset and last-obs logic of an MDP.
            assert(len(self.observations[0]) == len(self.rewards[0]) + 1)

            if ts >= num_timesteps:
                break

        # Batch all trajectories together along batch axis.
        return_obs = np.stack(return_obs, axis=0)
        return_actions = np.stack(return_actions, axis=0)
        return_rewards = np.stack(return_rewards, axis=0)
        return_terminateds = np.stack(return_terminateds, axis=0)
        return_truncateds = np.stack(return_truncateds, axis=0)
        return_initial_h = np.stack(return_initial_h, axis=0)
        return_next_obs = np.stack(return_next_obs, axis=0)

        if not self.continuous_episodes:
            return_masks = np.array(
                [
                    [1.0 if i < m else 0.0 for i in range(self.max_seq_len)]
                    for m in return_masks
                ],
                dtype=np.float32,
            )
            return return_obs, return_next_obs, return_actions, return_rewards, return_terminateds, return_truncateds, return_initial_h, return_masks
        else:
            return return_obs, return_next_obs, return_actions, return_rewards, return_terminateds, return_truncateds, return_initial_h

    def _split_by_env(self, inputs):
        return [inputs[i] for i in range(self.env.num_envs)]

    def _process_and_pad(self, inputs):
        # inputs=T x [dim]
        inputs = np.stack(inputs, axis=0)
        # inputs=[T, dim]
        inputs = np.pad(
            inputs,
            [(0, self.max_seq_len - inputs.shape[0])] + [(0, 0)] * (len(inputs.shape) - 1),
            "constant",
            constant_values=0.0,
        )
        # inputs=[[B=num_envs, Tmax, ... dims]
        return inputs


if __name__ == "__main__":
    config = (
        AlgorithmConfig()
        .environment("ALE/MsPacman-v5", env_config={
            # TODO: For now, do deterministic only.
            #  According to SimplE paper, stochastic envs should NOT be a problem, though.
            "frameskip": 4,
            # DreamerV3 paper does not specify, whether Atari100k is run
            # w/ or w/o sticky actions, just that frameskip=4.
            "repeat_action_probability": 0.25,
            "full_action_space": True,
        })
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=64)
    )
    env_runner = EnvRunner(
        model=None,
        config=config,
        max_seq_len=None,
        continuous_episodes=True,
        _debug_count_env=True,
    )
    for _ in range(100):
        obs, next_obs, actions, rewards, terminateds, truncateds, initial_h = (
            env_runner.sample(random_actions=True)
        )
        print("obs=", obs[:, :, 0,0])
        print("next_obs=", next_obs[:, 0,0])
        print("actions=", actions)
        print("terminateds=", terminateds)
        print("initial_h=", initial_h)
        print()
