from collections import deque
from typing import List, Union

import numpy as np

from utils.episode import Episode


class EpisodeReplayBuffer:
    """Buffer that stores (completed or truncated) episodes.
    """
    def __init__(self, capacity: int = 10000):
        # Max. num timesteps to store before ejecting old data.
        self.capacity = capacity

        # The actual episode buffer.
        self.episodes = deque()
        self.episode_id_to_index = {}
        self._num_episodes_ejected = 0

        # Deque storing all index tuples [eps-idx, pos-in-eps-idx].
        # We sample uniformly from the set of these indices in a `sample()`
        # call.
        self._indices = []

        self.size = 0

    def add(self, episodes: Union[List[Episode], Episode]):
        if isinstance(episodes, Episode):
            episodes = [episodes]

        for eps in episodes:
            self.size += len(eps)
            # Ongoing episode, concat to existing record.
            if eps.id_ in self.episode_id_to_index:
                buf_idx = self.episode_id_to_index[eps.id_]
                existing_eps = self.episodes[buf_idx - self._num_episodes_ejected]
                old_len = len(existing_eps)
                self._indices.extend([(buf_idx, old_len + i) for i in range(len(eps))])
                existing_eps.concat_episode(eps)
            # New episode. Add to end of our buffer.
            else:
                self.episodes.append(eps)
                buf_idx = len(self.episodes) - 1 + self._num_episodes_ejected
                self.episode_id_to_index[eps.id_] = buf_idx
                self._indices.extend([(buf_idx, i) for i in range(len(eps))])

            # Eject old records from front.
            while self.size > self.capacity:
                # Eject oldest episode.
                ejected_eps = self.episodes.popleft()
                self.size -= len(ejected_eps)
                # Erase from indices.
                ejected_idx = self.episode_id_to_index[ejected_eps.id_]
                del self.episode_id_to_index[ejected_eps.id_]
                new_indices = []
                idx_cursor = 0
                for i, idx_tuple in enumerate(self._indices):
                    if idx_cursor is not None and idx_tuple[0] == ejected_idx:
                        new_indices.extend(self._indices[idx_cursor:i])
                        idx_cursor = None
                    elif idx_cursor is None and idx_tuple[0] != ejected_idx:
                        idx_cursor = i
                if idx_cursor is not None:
                    new_indices.extend(self._indices[idx_cursor:])
                self._indices = new_indices
                self._num_episodes_ejected += 1

    def sample(self, batch_size_B: int = 16, batch_length_T: int = 64):
        # Uniformly sample n samples from self._indices.
        index_tuples_idx = np.random.randint(
            0, len(self._indices), size=batch_size_B * 10
        )
        observations = [[] for _ in range(batch_size_B)]
        actions = [[] for _ in range(batch_size_B)]
        rewards = [[] for _ in range(batch_size_B)]
        continues = [[] for _ in range(batch_size_B)]
        B = 0
        idx_cursor = 0
        while B < batch_size_B:
            index_tuple = self._indices[index_tuples_idx[idx_cursor]]
            episode_idx, episode_ts = (
                index_tuple[0] - self._num_episodes_ejected, index_tuple[1]
            )
            episode = self.episodes[episode_idx]
            if len(rewards[B]) == 0:
                if episode_ts == 0:
                    rewards[B].append(0.0)
                    continues[B].append(False)
                else:
                    rewards[B].append(episode.rewards[episode_ts - 1])
                    continues[B].append(True)
            else:
                episode_ts = 0

            observations[B].extend(episode.observations[episode_ts:-1])
            actions[B].extend(episode.actions[episode_ts:])
            rewards[B].extend(episode.rewards[episode_ts:])
            continues[B].extend([True] * (len(episode) - episode_ts))
            continues[B][-1] = False

            if len(observations[B]) > batch_length_T:
                observations[B] = observations[B][:batch_length_T]
                actions[B] = actions[B][:batch_length_T]
                rewards[B] = rewards[B][:batch_length_T]
                continues[B] = continues[B][:batch_length_T]
                B += 1
            idx_cursor += 1

        return (
            np.array(observations),
            np.array(actions),
            np.array(rewards),
            np.array(continues),
        )


if __name__ == "__main__":
    buffer = EpisodeReplayBuffer(capacity=10000)
    B = 10
    T = 4

    def _get_episode():
        eps = Episode(initial_observation=0.0)
        ts = np.random.randint(1, 500)
        for t in range(ts):
            eps.add_timestep(
                observation=float(t + 1),
                action=int(t + 1),
                reward=0.1 * (t + 1),
            )
        eps.is_terminated = np.random.random() > 0.5
        return eps

    for _ in range(200):
        episodes = _get_episode()
        buffer.add(episodes)

    for _ in range(1000):
        obs, actions, rewards, continues = buffer.sample(
            batch_size_B=16, batch_length_T=64
        )
        # All fields have same shape.
        assert obs.shape == rewards.shape == actions.shape == continues.shape
        # All rewards match obs, except when done.
        assert np.all(np.where(continues, np.equal(obs * 0.1, rewards), True))
        # All actions are always one larger than their obs, except when done.
        assert np.all(np.where(continues, np.equal(obs + 1.0, actions), True))
        # All ts=0 episode rewards should be 0.0 iff at beginning of a batch row.
        assert np.all(
            np.where(np.logical_not(continues[:, 0]), rewards[:, 0] == 0.0, True))
