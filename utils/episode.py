from typing import Optional
import uuid

import numpy as np


class Episode:

    def __init__(self, id_: Optional[str] = None, initial_observation=None):
        self.id_ = id_ or uuid.uuid4().hex
        # Observations: t0 (initial obs) to T.
        self.observations = [] if initial_observation is None else [initial_observation]
        # Actions: t1 to T.
        self.actions = []
        # Rewards: t1 to T.
        self.rewards = []
        # obs(T) is the final observation in the episode.
        self.is_terminated = False

    def concat_episode(self, episode_chunk: "Episode"):
        assert episode_chunk.id_ == self.id_
        assert self.is_terminated is False

        episode_chunk.validate()

        # Make sure, end matches other episode chunk's beginning.
        assert np.all(episode_chunk.observations[0] == self.observations[-1])
        # Pop out our end.
        self.observations.pop()

        # Extend ourselves.
        self.observations.extend(episode_chunk.observations)
        self.actions.extend(episode_chunk.actions)
        self.rewards.extend(episode_chunk.rewards)

        if episode_chunk.is_terminated:
            self.is_terminated = True
        # Validate.
        self.validate()

    def add_timestep(self, observation, action, reward, is_terminated=False):
        assert not self.is_terminated

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.is_terminated = is_terminated
        self.validate()

    def add_initial_observation(self, initial_observation):
        assert not self.is_terminated
        assert len(self.observations) == 0

        self.observations.append(initial_observation)
        self.validate()

    def validate(self):
        # Make sure we always have one more obs stored than rewards (and actions)
        # due to the reset and last-obs logic of an MDP.
        assert (
            len(self.observations) == len(self.rewards) + 1 == len(self.actions) + 1
        )

    def get_return(self):
        return sum(self.rewards)

    def __len__(self):
        assert len(self.observations) > 0, (
            "ERROR: Cannot determine length of episode that hasn't started yet! "
            "Call `Episode.add_initial_obs(initial_observation=...)` first "
            "(after which `len(Episode)` will be 0)."
        )
        return len(self.observations) - 1
