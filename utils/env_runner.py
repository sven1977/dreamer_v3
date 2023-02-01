import gymnasium as gym
import numpy as np
import tensorflow as tf
from typing import Optional

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.sample_batch import MultiAgentBatch


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
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
            )
        else:
            self.env = gym.vector.make(
                self.config.env,
                num_envs=self.config.num_envs_per_worker,
                asynchronous=self.config.remote_worker_envs,
            )
        self.needs_initial_reset = True
        self.observations = [[] for _ in range(self.env.num_envs)]
        #self.next_observations = []
        self.actions = [[] for _ in range(self.env.num_envs)]
        self.rewards = [[] for _ in range(self.env.num_envs)]
        self.episode_lengths = [0, 0]

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
        ts = 0
        ts_sequences = [len(self.observations[i]) for i in range(self.env.num_envs)]
        return_obs = []
        #return_next_obs = []
        return_actions = []
        return_rewards = []
        return_masks = []

        if force_reset or self.needs_initial_reset:
            obs, _ = self.env.reset()
            self.needs_initial_reset = False
            for i, o in enumerate(self._split_by_env(obs)):
                self.observations[i].append(o)

        while True:
            if random_actions:
                actions = self.env.action_space.sample()
            else:
                action_logits = self.model(obs)
                # Sample.
                if explore:
                    actions = tf.random.categorical(action_logits, num_samples=1)
                # Greedy.
                else:
                    actions = np.argmax(action_logits, axis=-1)

            obs, rewards, terminateds, truncateds, infos = self.env.step(actions)

            #self.next_observations.append(next_obs)
            for i, a in enumerate(self._split_by_env(actions)):
                self.actions[i].append(a)
            for i, r in enumerate(self._split_by_env(rewards)):
                self.rewards[i].append(r)

            #obs = next_obs

            ts += self.env.num_envs
            for i in range(self.env.num_envs):
                ts_sequences[i] += 1
                if terminateds[i] or truncateds[i] or ts_sequences[i] == self.max_seq_len:
                    seq_len = ts_sequences[i]
                    ts_sequences[i] = 0
                    if terminateds[i] or truncateds[i]:
                        self.observations[i].append(infos["final_observation"][i])

                    return_obs.append(self._process_and_pad(self.observations[i]))
                    #return_next_obs.append(self._process_and_pad(self.next_observations))
                    return_actions.append(self._process_and_pad(self.actions[i]))
                    return_rewards.append(self._process_and_pad(self.rewards[i]))
                    return_masks.append(seq_len)

                    self.observations[i] = []
                    #self.next_observations = []
                    self.actions[i] = []
                    self.rewards[i] = []

            for i, o in enumerate(self._split_by_env(obs)):
                self.observations[i].append(o)

            if ts >= num_timesteps:
                break

        # Batch all trajectories together along batch axis.
        return_obs = np.stack(return_obs, axis=0)
        #return_next_obs = np.concatenate(return_next_obs, axis=0)
        return_actions = np.stack(return_actions, axis=0)
        return_rewards = np.stack(return_rewards, axis=0)
        return_masks = np.array([[1.0 if i < m else 0.0 for i in range(self.max_seq_len)] for m in return_masks])

        return return_obs, return_actions, return_rewards, return_masks

    def _split_by_env(self, inputs):
        return [inputs[i] for i in range(self.env.num_envs)]

    def _process_and_pad(self, inputs):
        # inputs=T x [dim]
        inputs = np.stack(inputs, axis=0)
        # inputs=[T, dim]

        if len(inputs.shape) == 2:
            inputs = np.pad(inputs, ((0, self.max_seq_len - inputs.shape[0]), (0, 0)), "constant", constant_values=0.0)
        else:
            inputs = np.pad(
                inputs,
                (0, self.max_seq_len - inputs.shape[0],),
                "constant",
                constant_values=0.0,
            )
        # inputs=[[B (num_envs), Tmax, dim]
        return inputs


if __name__ == "__main__":
    #gym.register("atari", lambda: gym.wrappers.EnvCompatibility(gym.make("GymV26Environment-v0", env_id="ALE/MsPacman-v5")))
    config = (
        AlgorithmConfig()
        .environment("ALE/MsPacman-v5")
        .rollouts(num_envs_per_worker=2, rollout_fragment_length=200)
    )
    env_runner = EnvRunner(model=None, config=config, max_seq_len=64)
    print(env_runner.sample(random_actions=True))
