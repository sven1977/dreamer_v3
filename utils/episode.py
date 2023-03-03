from typing import Optional
import uuid

import numpy as np


class Episode:

    def __init__(self, id_: Optional[str] = None, *, initial_observation=None, initial_h_state=None, initial_render_image=None):
        self.id_ = id_ or uuid.uuid4().hex
        # Observations: t0 (initial obs) to T.
        self.observations = [] if initial_observation is None else [initial_observation]
        # Actions: t1 to T.
        self.actions = []
        # Rewards: t1 to T.
        self.rewards = []
        # h-states: t0 (in case this Episode is a continuation chunk, we need to know
        # about the initial h) to T.
        self.h_states = [] if initial_h_state is None else [initial_h_state]
        # obs(T) is the final observation in the episode.
        self.is_terminated = False
        # RGB uint8 images from rendering the env; the images include the corresponding
        # rewards.
        assert initial_render_image is None or initial_observation is not None
        self.render_images = [] if initial_render_image is None else [initial_render_image]

    def concat_episode(self, episode_chunk: "Episode"):
        assert episode_chunk.id_ == self.id_
        assert not self.is_terminated

        episode_chunk.validate()

        # Make sure, end matches other episode chunk's beginning.
        assert np.all(episode_chunk.observations[0] == self.observations[-1])
        # Pop out our end.
        self.observations.pop()
        if len(self.h_states) > 0:
            self.h_states.pop()

        # Extend ourselves. In case, episode_chunk is already terminated (and numpyfied)
        # we need to convert to lists (as we are ourselves still filling up lists).
        self.observations.extend(list(episode_chunk.observations))
        self.actions.extend(list(episode_chunk.actions))
        self.rewards.extend(list(episode_chunk.rewards))
        self.h_states.extend(list(episode_chunk.h_states))

        if episode_chunk.is_terminated:
            self.is_terminated = True
        # Validate.
        self.validate()

    def add_timestep(self, observation, action, reward, *,
                     h_state=None, is_terminated=False, render_image=None):
        assert not self.is_terminated

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        if h_state is not None:
            self.h_states.append(h_state)
        if render_image is not None:
            self.render_images.append(render_image)
        self.is_terminated = is_terminated
        self.validate()

    def add_initial_observation(self, *, initial_observation, initial_h_state=None, initial_render_image=None):
        assert not self.is_terminated
        assert len(self.observations) == 0

        self.observations.append(initial_observation)
        if initial_h_state is not None:
            self.h_states.append(initial_h_state)
        if initial_render_image is not None:
            self.render_images.append(initial_render_image)
        self.validate()

    def validate(self):
        # Make sure we always have one more obs stored than rewards (and actions)
        # due to the reset and last-obs logic of an MDP.
        assert (
            len(self.observations) == len(self.rewards) + 1 == len(self.actions) + 1
        )
        # H-states are either non-existent OR we have the same as rewards.
        assert len(self.h_states) == 0 or len(self.h_states) == len(self.observations)
        # Render images are either non-existent OR we have the same as observations.
        #assert len(self.render_images) == 0 or len(self.render_images) == len(self.observations)

        # Convert all lists to numpy arrays, if we are terminated.
        if self.is_terminated:
            self.observations = np.array(self.observations)
            self.actions = np.array(self.actions)
            self.rewards = np.array(self.rewards)
            self.h_states = np.array(self.h_states)
            self.render_images = np.array(self.render_images, dtype=np.uint8)

    def get_return(self):
        return sum(self.rewards)

    def __len__(self):
        assert len(self.observations) > 0, (
            "ERROR: Cannot determine length of episode that hasn't started yet! "
            "Call `Episode.add_initial_obs(initial_observation=...)` first "
            "(after which `len(Episode)` will be 0)."
        )
        return len(self.observations) - 1
