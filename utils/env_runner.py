from functools import partial
import gymnasium as gym
import numpy as np
from supersuit.generic_wrappers import resize_v1
import tensorflow as tf
from typing import Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.sample_batch import MultiAgentBatch


class CountEnv(gym.ObservationWrapper):
    def reset(self, **kwargs):
        self.__counter = 0
        return super().reset(**kwargs)

    def observation(self, observation):
        observation[0][0][0] = self.__counter
        self.__counter += 1
        return observation


class EnvRunner:
    """An environment runner to locally collect data from vectorized gym environments.
    """

    def __init__(self, model, config: AlgorithmConfig, max_seq_len: int = 100):
        """Initializes an EnvRunner instance.

        Args:
            config: The config to use to setup this EnvRunner.
        """
        self.model = model
        self.config = config
        self.max_seq_len = max_seq_len
        if self.config.env.startswith("ALE"):
            self.env = gym.vector.make(
                "GymV26Environment-v0",
                env_id=self.config.env,
                wrappers=[partial(resize_v1, x_size=64, y_size=64)],#, CountEnv],
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
                make_kwargs=self.config.env_config,
            )
        else:
            self.env = gym.vector.make(
                self.config.env,
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
                make_kwargs=self.config.env_config,
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
        return_next_obs = []
        return_actions = []
        return_rewards = []
        return_terminateds = []
        return_truncateds = []
        return_initial_h = []
        return_masks = []

        if force_reset or self.needs_initial_reset:
            obs, _ = self.env.reset()
            self.next_h_states = self.model._get_initial_h(
                batch_size=self.config.num_envs_per_worker
            ).numpy()
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
                self.next_h_states = (
                    self.model.forward_inference(obs, actions, tf.convert_to_tensor(self.next_h_states))
                ).numpy()
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
                if terminateds[i] or truncateds[i] or ts_sequences[i] == self.max_seq_len:
                    seq_len = ts_sequences[i]
                    ts_sequences[i] = 0

                    return_obs.append(self._process_and_pad(self.observations[i][:-1]))
                    return_actions.append(self._process_and_pad(self.actions[i]))
                    return_rewards.append(self._process_and_pad(self.rewards[i]))
                    return_terminateds.append(self._process_and_pad(self.terminateds[i]))
                    return_truncateds.append(self._process_and_pad(self.truncateds[i]))
                    return_initial_h.append(self.current_sequence_initial_h[i].copy())
                    return_masks.append(seq_len)

                    # The last entry in self.observations[i] is already the reset obs
                    # of the new episode.
                    if terminateds[i] or truncateds[i]:
                        return_next_obs.append(infos["final_observation"][i])
                        # Reset h-states to all zeros b/c we are starting a new episode.
                        self.next_h_states[i] = self.model._get_initial_h(batch_size=0).numpy()
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
        return_next_obs = np.stack(return_next_obs, axis=0)
        return_actions = np.stack(return_actions, axis=0)
        return_rewards = np.stack(return_rewards, axis=0)
        return_terminateds = np.stack(return_terminateds, axis=0)
        return_truncateds = np.stack(return_truncateds, axis=0)
        return_initial_h = np.stack(return_initial_h, axis=0)
        return_masks = np.array(
            [
                [1.0 if i < m else 0.0 for i in range(self.max_seq_len)]
                for m in return_masks
            ],
            dtype=np.float32,
        )

        return return_obs, return_next_obs, return_actions, return_rewards, return_terminateds, return_truncateds, return_initial_h, return_masks

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
    #gym.register("atari", lambda: gym.wrappers.EnvCompatibility(gym.make("GymV26Environment-v0", env_id="ALE/MsPacman-v5")))
    config = (
        AlgorithmConfig()
        .environment("ALE/MsPacman-v5", env_config={
            # TODO: For now, do deterministic only.
            #  According to SimplE paper, stochastic envs should NOT be a problem, though.
            "frameskip": 4,
            # DreamerV3 paper does not specify, whether Atari100k is run
            # w/ or w/o sticky actions, just that frameskip=4.
            "repeat_action_probability": 0.0,
        })
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=200)
    )
    env_runner = EnvRunner(model=None, config=config, max_seq_len=64)
    for _ in range(100):
        obs, next_obs, actions, rewards, terminateds, truncateds, mask = (
            env_runner.sample(random_actions=True)
        )
        print(obs.shape) # obs shape
        print(obs[:, :, 0,0,0])
        print(next_obs[:, 0,0,0])
        print(actions)
        print(terminateds)
        print(mask)
