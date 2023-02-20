"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
from functools import partial
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from supersuit.generic_wrappers import resize_v1, color_reduction_v0
import tensorflow as tf

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.wrappers.atari_wrappers import NoopResetEnv, MaxAndSkipEnv

from utils.episode import Episode


class CountEnv(gym.ObservationWrapper):
    def reset(self, **kwargs):
        self.__counter = 0
        return super().reset(**kwargs)

    def observation(self, observation):
        # For gray-scaled observations.
        observation[0][0] = self.__counter
        # For 3-color observations.
        #observation[0][0][0] = self.__counter__
        self.__counter += 1
        return observation


class EnvRunnerV2:
    """An environment runner to locally collect data from vectorized gym environments.
    """

    def __init__(
        self,
        model,
        config: AlgorithmConfig,
        _debug_count_env=False
    ):
        """Initializes an EnvRunner instance.

        Args:
            config: The config to use to setup this EnvRunner.
        """
        self.model = model
        self.config = config

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
        self.num_envs = self.env.num_envs
        assert self.num_envs == self.config.num_envs_per_worker

        self.needs_initial_reset = True
        self.episodes = [None for _ in range(self.num_envs)]
        self.done_episodes_for_metrics = []
        self.next_h_states = None
        #self.current_sequence_initial_h = None

        # The currently ongoing episodes' returns (sum of rewards).
        #self.episode_returns = [0.0 for _ in range(self.num_envs)]
        # The returns of already finished episodes, ready to be collected by a call
        # to `get_metrics()`.
        #self.episode_returns_to_collect = []

    def sample(self, explore: bool = True, random_actions: bool = False):
        if self.config.batch_mode == "complete_episodes":
            raise NotImplementedError
        else:
            return self.sample_timesteps(
                num_timesteps=(
                    self.config.rollout_fragment_length
                    * self.num_envs
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
    ) -> Tuple[List[Episode], List[Episode]]:
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
        done_episodes_to_return = []

        if force_reset or self.needs_initial_reset:
            obs, _ = self.env.reset()

            self.episodes = [Episode() for _ in range(self.num_envs)]

            # Get initial states for all tracks.
            if self.model is not None:
                self.next_h_states = self.model._get_initial_h(
                    batch_size=self.num_envs
                ).numpy()
            else:
                self.next_h_states = np.array([0.0] * self.num_envs)

            #self.current_sequence_initial_h = self.next_h_states.copy()
            self.needs_initial_reset = False

            for i, o in enumerate(self._split_by_env(obs)):
                self.episodes[i].add_initial_observation(initial_observation=o)
        else:
            obs = np.stack([eps.observations[-1] for eps in self.episodes])

        ts = 0
        #ts_sequences = [len(eps) - 1 for eps in self.episodes]

        while True:
            if random_actions:
                # TODO: hack; right now, our model (a world model) does not have an
                #  actor head yet. Still perform a forward pass to get the next h-states.
                actions = self.env.action_space.sample()
                if self.model is not None:
                    self.next_h_states = (
                        self.model.world_model.forward_inference(
                            obs,
                            actions,
                            tf.convert_to_tensor(self.next_h_states),
                        )
                    ).numpy()
                else:
                    self.next_h_states = np.array([1.0] * self.num_envs)
            else:
                # Sample.
                if explore:
                    a, h = self.model(
                        obs,
                        initial_h=tf.convert_to_tensor(self.next_h_states),
                    )
                    actions = a.numpy()
                    self.next_h_states = h.numpy()
                # Greedy.
                else:
                    raise NotImplementedError
                    #actions = np.argmax(action_logits, axis=-1)

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)
            ts += self.num_envs

            for i, (o, a, r, term, trunc) in enumerate(zip(
                self._split_by_env(obs),
                self._split_by_env(actions),
                self._split_by_env(rewards),
                self._split_by_env(terminateds),
                self._split_by_env(truncateds),
            )):
                # The last entry in self.observations[i] is already the reset
                # obs of the new episode.
                if term or trunc:
                    self.episodes[i].add_timestep(
                        infos["final_observation"][i],
                        a,
                        r,
                        is_terminated=True,
                    )
                    # Reset h-states to all zeros b/c we are starting a new episode.
                    if self.model is not None:
                        self.next_h_states[i] = self.model._get_initial_h(
                            batch_size=0).numpy()
                    else:
                        self.next_h_states[i] = 0.0

                    done_episodes_to_return.append(self.episodes[i])

                    self.episodes[i] = Episode(initial_observation=o)
                else:
                    self.episodes[i].add_timestep(o, a, r, term or trunc)

            if ts >= num_timesteps:
                break

        # Return done episodes ...
        self.done_episodes_for_metrics.extend(done_episodes_to_return)
        # ... and all ongoing episode chunks. Also, make sure, we return
        # a copy and start new chunks so that callers of this function
        # don't alter our Episode objects.
        ongoing_episodes = self.episodes
        self.episodes = [
            Episode(id_=eps.id_, initial_observation=eps.observations[-1])
            for eps in self.episodes
        ]

        return done_episodes_to_return, ongoing_episodes

    def get_metrics(self):
        metrics = {
            "episode_returns": self.done_episodes_for_metrics.get_return(),
        }
        self.done_episodes_for_metrics.clear()
        return metrics

    def _split_by_env(self, inputs):
        return [inputs[i] for i in range(self.num_envs)]


if __name__ == "__main__":
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
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=64)
    )
    env_runner = EnvRunnerV2(
        model=None,
        config=config,
        _debug_count_env=True,
    )
    for _ in range(10):
        done_episodes, ongoing_episodes = (
            env_runner.sample(random_actions=True)
        )
        for eps in done_episodes:
            assert eps.is_terminated
            print(f"done episode {eps.id_} obs[0]={eps.observations[0][0][0]} obs[-1]={eps.observations[-1][0][0]}")
        for eps in ongoing_episodes:
            assert not eps.is_terminated
            print(f"ongoing episode {eps.id_} obs[0]={eps.observations[0][0][0]} obs[-1]={eps.observations[-1][0][0]}")
        print()

    #obs, next_obs, actions, rewards, terminateds, truncateds = (
    #    env_runner.sample_episodes(num_episodes=10, random_actions=True)
    #)
    #mean_episode_return = np.mean([np.sum(rets) for rets in rewards])
    #print(len(obs))
    #print(f"mean(R)={mean_episode_return}")