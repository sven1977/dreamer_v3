from collections import deque
import copy
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
        # Maps (unique) episode IDs to the idex under which to find this episode
        # within our `self.episodes` Deque. Note that after ejection started, the
        # indices will NOT be changed. We will therefore need to offset these indices
        # by the number of episodes that have already been ejected.
        self.episode_id_to_index = {}
        # The number of episodes that have already been ejected from the buffer
        # due to reaching capacity. This is the offset, which we have to subtract
        # from the episode index to get the actual location within `self.episodes`.
        self._num_episodes_ejected = 0

        # Deque storing all index tuples [eps-idx, pos-in-eps-idx].
        # We sample uniformly from the set of these indices in a `sample()`
        # call.
        self._indices = []

        # The size of the buffer in timesteps.
        self.size = 0

    def add(self, episodes: Union[List[Episode], Episode]):
        if isinstance(episodes, Episode):
            episodes = [episodes]

        for eps in episodes:
            # Make sure we don't change what's coming in from the user.
            eps = copy.deepcopy(eps)

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

            # Eject old records from front of deque.
            while self.size > self.capacity:
                # Eject oldest episode.
                ejected_eps = self.episodes.popleft()
                ejected_eps_len = len(ejected_eps)
                # Correct our size.
                self.size -= len(ejected_eps)
                # Erase episode from all our indices.
                # Main episode index.
                ejected_idx = self.episode_id_to_index[ejected_eps.id_]
                del self.episode_id_to_index[ejected_eps.id_]
                # All timestep indices that this episode owned.
                new_indices = []
                idx_cursor = 0
                for i, idx_tuple in enumerate(self._indices):
                    if idx_cursor is not None and idx_tuple[0] == ejected_idx:
                        new_indices.extend(self._indices[idx_cursor:i])
                        idx_cursor = None
                    elif idx_cursor is None:
                        if idx_tuple[0] != ejected_idx:
                            idx_cursor = i
                        # Early-out: We reached the end of the to-be-ejected episode.
                        # We can stop searching further here.
                        elif idx_tuple[1] == ejected_eps_len - 1:
                            assert self._indices[i+1][0] != idx_tuple[0]
                            idx_cursor = i + 1
                            break
                if idx_cursor is not None:
                    new_indices.extend(self._indices[idx_cursor:])
                self._indices = new_indices
                # Increase episode ejected counter.
                self._num_episodes_ejected += 1

    def sample(self, batch_size_B: int = 16, batch_length_T: int = 64):
        # Uniformly sample n samples from self._indices.
        index_tuples_idx = []
        observations = [[] for _ in range(batch_size_B)]
        actions = [[] for _ in range(batch_size_B)]
        rewards = [[] for _ in range(batch_size_B)]
        is_first = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_last = [[False] * batch_length_T for _ in range(batch_size_B)]
        is_terminated = [[False] * batch_length_T for _ in range(batch_size_B)]
        #h_states = [[] for _ in range(batch_size_B)]
        # Sampled indices. Each index is a tuple: episode-idx + ts-idx.
        #indices = [[] for _ in range(batch_size_B)]

        B = 0
        T = 0
        idx_cursor = 0
        #episode_h_states = False
        while B < batch_size_B:
            # Ran out of uniform samples -> Sample new set.
            if len(index_tuples_idx) <= idx_cursor:
                index_tuples_idx.extend(list(np.random.randint(
                    0, len(self._indices), size=batch_size_B * 10
                )))

            index_tuple = self._indices[index_tuples_idx[idx_cursor]]
            episode_idx, episode_ts = (
                index_tuple[0] - self._num_episodes_ejected, index_tuple[1]
            )
            episode = self.episodes[episode_idx]
            #episode_len = len(episode)
            #episode_h_states = len(episode.h_states) > 0

            # Starting a new chunk, set continue to False.
            is_first[B][T] = True

            # Begin of new batch item (row).
            if len(rewards[B]) == 0:
                # And we are at the start of an episode: Set reward to 0.0.
                if episode_ts == 0:
                    rewards[B].append(0.0)
                    #if episode_h_states:
                    #    h_states[B].append(np.zeros_like(episode.h_states[0]))
                    #continues[B].append(False)
                # We are in the middle of an episode: Set reward and h_state to
                # the previous timestep's values.
                else:
                    rewards[B].append(episode.rewards[episode_ts - 1])
                    #if episode_h_states:
                    #    h_states[B].append(episode.h_states[episode_ts - 1])
                    #continues[B].append(True)
            # We are in the middle of a batch item (row). Concat next episode to this
            # row from the episode's beginning. In other words, we never concat
            # a middle of an episode to another truncated one.
            else:
                episode_ts = 0
                rewards[B].append(0.0)

            observations[B].extend(episode.observations[episode_ts:])
            # Repeat last action to have the same number of actions than observations.
            actions[B].extend(episode.actions[episode_ts:] + [episode.actions[-1]])
            # Number of rewards are also the same as observations b/c we have the
            # initial 0.0 one.
            rewards[B].extend(episode.rewards[episode_ts:])
            #if episode_h_states:
            #    h_states[B].extend(episode.h_states[episode_ts:])
            #continues[B].extend([True] * (episode_len - episode_ts))
            #continues[B][-1] = False
            #indices[B].extend([[index_tuple[0], episode_ts + i] for i in range(episode_len - episode_ts)])

            T = min(len(observations[B]), batch_length_T)

            # Set is_last=True.
            is_last[B][T-1] = True
            # If episode is terminated and we have reached the end of it, set
            # is_terminated=True.
            if episode.is_terminated and T == len(observations[B]):
                is_terminated[B][T-1] = True

            # We are done with this batch row.
            if T == batch_length_T:
                # We may have overfilled this row: Clip trajectory at the end.
                observations[B] = observations[B][:batch_length_T]
                actions[B] = actions[B][:batch_length_T]
                rewards[B] = rewards[B][:batch_length_T]
                #if episode_h_states:
                #    h_states[B] = h_states[B][:batch_length_T]
                #continues[B] = continues[B][:batch_length_T]
                #indices[B] = indices[B][:batch_length_T]
                # Start filling the next row.
                B += 1
                T = 0

            # Use next sampled episode/ts pair to fill the row.
            idx_cursor += 1

        ret = {
            "obs": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "is_first": np.array(is_first),
            "is_last": np.array(is_last),
            "is_terminated": np.array(is_terminated),
            #"continues": np.array(continues),
            #"indices": np.array(indices),
        }
        #if episode_h_states:
        #    ret["h_states"] = np.array(h_states)

        return ret

    #def update_h_states(self, h_states, indices):
    #    # Loop through batch items (rows).
    #    for i, idxs in enumerate(indices):
    #        # Loop through timesteps.
    #        for j, index_tuple in enumerate(idxs):
    #            # Find the correct episode and the timestep therein and update
    #            # the h-value at that very position.
    #            episode_idx, episode_ts = (
    #                index_tuple[0] - self._num_episodes_ejected, index_tuple[1]
    #            )
    #            episode = self.episodes[episode_idx]
    #            episode.h_states[episode_ts] = h_states[i][j]

    def get_num_episodes(self):
        return len(self.episodes)

    def get_num_timesteps(self):
        return len(self._indices)


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
                action=int(t),
                reward=0.1 * (t + 1),
            )
        eps.is_terminated = np.random.random() > 0.5
        return eps

    for _ in range(200):
        episodes = _get_episode()
        buffer.add(episodes)

    for _ in range(1000):
        sample = buffer.sample(
            batch_size_B=16, batch_length_T=64
        )
        obs, actions, rewards, is_first, is_last, is_terminated = (
            sample["obs"], sample["actions"], sample["rewards"],
            sample["is_first"], sample["is_last"], sample["is_terminated"]
        )
        # Make sure, is_first and is_last are trivially correct.
        assert np.all(is_last[:, -1])
        assert np.all(is_first[:, 0])

        # All fields have same shape.
        assert obs.shape == rewards.shape == actions.shape == is_first.shape == is_last.shape == is_terminated.shape

        # All rewards match obs.
        assert np.all(np.equal(obs * 0.1, rewards))
        # All actions are always the same as their obs, except when terminated (one
        # less).
        assert np.all(np.where(is_last, True, np.equal(obs, actions)))
        # All actions on is_terminated=True must be the same as the previous ones
        # (we repeat the action as the last one is a dummy one anyways (action
        # picked in terminal observation/state)).
        assert np.all(np.where(is_terminated[:, 1:], np.equal(actions[:,1:], actions[:,:-1]), True))
        # Where is_terminated, the next rewards should always be 0.0 (reset rewards).
        assert np.all(np.where(is_terminated[:, :-1], rewards[:, 1:] == 0.0, True))
