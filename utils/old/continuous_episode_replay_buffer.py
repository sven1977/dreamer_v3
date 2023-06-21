from collections import deque

import numpy as np


class ContinuousEpisodeReplayBuffer:
    """Buffer that stores N episodes back-to-back per "batch-track".

    Data passed into `add()` must be of shape: (B, ...), where B is a fixed, hard-coded
    value that needs to be known at construction time.

    sample()
    """

    def __init__(self, capacity: int = 10000, num_data_tracks: int = 1):
        self.capacity = capacity
        self.num_data_tracks = num_data_tracks

        # Max length of each batch-track. Episodes and episode fragments coming
        # in through `add()` are added to either one of these tracks, depending on
        # their batch index in the incoming data.
        self.data_track_size = self.capacity // self.num_data_tracks

        self.buffers = {}
        for key in [
            "obs",
            "next_obs",
            "actions",
            "rewards",
            "terminateds",
            "truncateds",
            "h_states",
        ]:
            self.buffers[key] = self._make_buffer()

    def add(self, data: dict):
        for key, value in data.items():
            assert len(value) % self.num_data_tracks == 0
            # Split along batch axis.
            for i in range(len(value)):
                buf_idx = i % self.num_data_tracks
                self.buffers[key][buf_idx].append(value[i])

    def sample(self, num_items: int = 1):
        assert num_items % self.num_data_tracks == 0
        max_idx = len(self) // self.num_data_tracks
        indices = [
            np.random.randint(0, max_idx, size=num_items // self.num_data_tracks)
            for _ in range(self.num_data_tracks)
        ]
        sample = {}
        for key, bufs in self.buffers.items():
            sample[key] = np.stack(
                [
                    bufs[buf_idx][i]
                    for buf_idx in range(self.num_data_tracks)
                    for i in indices[buf_idx]
                ],
                axis=0,
            )
        return sample

    def save(self):
        np.savez(
            "buffer.npz",
            {key: np.stack(deq, axis=0) for key, deq in self.buffers.items()},
        )

    def load(self):
        buffer_content = np.load("buffer.npz")
        for key, value in buffer_content.items():
            self.buffers[key].clear()
            for row in list(value):
                self.buffers[key].append(row)

    def __len__(self):
        return len(self.buffers["obs"][0]) * self.num_data_tracks

    def _make_buffer(self):
        return [deque(maxlen=self.data_track_size) for _ in range(self.num_data_tracks)]


if __name__ == "__main__":
    buffer = ContinuousEpisodeReplayBuffer(capacity=100, num_data_tracks=2)
    B = 10
    T = 4

    def _get_batch():
        return {
            "obs": np.random.random(size=(B, T, 64, 64, 3)),
            "next_obs": np.random.random(size=(B, 64, 64, 3)),
            "actions": np.random.random(size=(B, T)),
            "rewards": np.random.random(size=(B, T)),
            "terminateds": np.random.random(size=(B, T)),
            "truncateds": np.random.random(size=(B, T)),
            "h_states": np.random.random(size=(B, 256)),
        }

    for _ in range(5):
        batch = _get_batch()
        buffer.add(batch)

    print(buffer.sample(2)["obs"].shape)
    print(buffer.sample(10)["obs"].shape)
