import uuid
from typing import Optional

import numpy as np


class Episode:
    def __init__(
        self,
        id_: Optional[str] = None,
        *,
        observations=None,
        actions=None,
        rewards=None,
        states=None,
        is_terminated=False,
        render_images=None,
    ):
        self.id_ = id_ or uuid.uuid4().hex
        # Observations: t0 (initial obs) to T.
        self.observations = [] if observations is None else observations
        # Actions: t1 to T.
        self.actions = [] if actions is None else actions
        # Rewards: t1 to T.
        self.rewards = [] if rewards is None else rewards
        # h-states: t0 (in case this Episode is a continuation chunk, we need to know
        # about the initial h) to T.
        self.states = states
        # obs(T) is the final observation in the episode.
        self.is_terminated = is_terminated
        # RGB uint8 images from rendering the env; the images include the corresponding
        # rewards.
        assert render_images is None or observations is not None
        self.render_images = [] if render_images is None else render_images

    def concat_episode(self, episode_chunk: "Episode"):
        assert episode_chunk.id_ == self.id_
        assert not self.is_terminated

        episode_chunk.validate()

        # Make sure, end matches other episode chunk's beginning.
        assert np.all(episode_chunk.observations[0] == self.observations[-1])
        # Pop out our end.
        self.observations.pop()
        # if len(self.states) > 0:
        #    self.states.pop()

        # Extend ourselves. In case, episode_chunk is already terminated (and numpyfied)
        # we need to convert to lists (as we are ourselves still filling up lists).
        self.observations.extend(list(episode_chunk.observations))
        self.actions.extend(list(episode_chunk.actions))
        self.rewards.extend(list(episode_chunk.rewards))
        self.states = episode_chunk.states  # .extend(list(episode_chunk.states))

        if episode_chunk.is_terminated:
            self.is_terminated = True
        # Validate.
        self.validate()

    def add_timestep(
        self,
        observation,
        action,
        reward,
        *,
        state=None,
        is_terminated=False,
        render_image=None,
    ):
        assert not self.is_terminated

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        # if state is not None:
        #    self.states.append(state)
        self.states = state
        if render_image is not None:
            self.render_images.append(render_image)
        self.is_terminated = is_terminated
        self.validate()

    def add_initial_observation(
        self, *, initial_observation, initial_state=None, initial_render_image=None
    ):
        assert not self.is_terminated
        assert len(self.observations) == 0

        self.observations.append(initial_observation)
        # if initial_state is not None:
        #    self.states.append(initial_state)
        self.states = initial_state
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)
        self.validate()

    def validate(self):
        # Make sure we always have one more obs stored than rewards (and actions)
        # due to the reset and last-obs logic of an MDP.
        assert len(self.observations) == len(self.rewards) + 1 == len(self.actions) + 1
        ## H-states are either non-existent OR we have the same as rewards.
        # assert len(self.states) == 0 or len(self.states) == len(self.observations)
        # Render images are either non-existent OR we have the same as observations.
        # assert len(self.render_images) == 0 or len(self.render_images) == len(self.observations)

        # Convert all lists to numpy arrays, if we are terminated.
        if self.is_terminated:
            self.observations = np.array(self.observations)
            self.actions = np.array(self.actions)
            self.rewards = np.array(self.rewards)
            # self.states = np.array(self.states)
            self.render_images = np.array(self.render_images, dtype=np.uint8)

    def get_return(self):
        return sum(self.rewards)

    def get_state(self):
        return list(
            {
                "id_": self.id_,
                "observations": self.observations,
                "actions": self.actions,
                "rewards": self.rewards,
                "states": self.states,
                "is_terminated": self.is_terminated,
            }.items()
        )

    @staticmethod
    def from_state(state):
        eps = Episode(id_=state[0][1])
        eps.observations = state[1][1]
        eps.actions = state[2][1]
        eps.rewards = state[3][1]
        eps.states = state[4][1]
        eps.is_terminated = state[5][1]
        return eps

    def __len__(self):
        assert len(self.observations) > 0, (
            "ERROR: Cannot determine length of episode that hasn't started yet! "
            "Call `Episode.add_initial_obs(initial_observation=...)` first "
            "(after which `len(Episode)` will be 0)."
        )
        return len(self.observations) - 1
