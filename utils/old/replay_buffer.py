from collections import deque
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffers = {
            "obs": deque(maxlen=capacity),
            "next_obs": deque(maxlen=capacity),
            "actions": deque(maxlen=capacity),
            "rewards": deque(maxlen=capacity),
            "terminateds": deque(maxlen=capacity),
            "truncateds": deque(maxlen=capacity),
            "mask": deque(maxlen=capacity),
            "h_states": deque(maxlen=capacity),
        }

    def add(self, data: dict):
        for key, value in data.items():
            # Split along batch axis.
            for i in range(len(value)):
                self.buffers[key].append(value[i])

    def sample(self, num_items: int = 1):
        indices = np.random.random_integers(0, len(self) - 1, size=num_items)
        sample = {}
        for key, buf in self.buffers.items():
            sample[key] = np.stack([buf[i] for i in indices], axis=0)
        return sample

    def save(self):
        np.savez("buffer.npz", {
            key: np.stack(deq, axis=0)
            for key, deq in self.buffers.items()
        })

    def load(self):
        buffer_content = np.load("buffer.npz")
        for key, value in buffer_content.items():
            self.buffers[key].clear()
            for row in list(value):
                self.buffers[key].append(row)

    def __len__(self):
        return len(self.buffers["obs"])


if __name__ == "__main__":
    buffer = ReplayBuffer(capacity=100)
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
            "mask": np.random.random(size=(B, T)),
            "h_states": np.random.random(size=(B, 256)),
        }

    for _ in range(5):
        batch = _get_batch()
        buffer.add(batch)

    print(buffer.sample()["obs"].shape)
    print(buffer.sample(2)["obs"].shape)
    print(buffer.sample(10)["obs"].shape)
