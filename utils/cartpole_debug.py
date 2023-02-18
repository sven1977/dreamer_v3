import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import tensorflow as tf


class CartPoleDebug(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        low = np.concatenate([np.array([0.0]), self.observation_space.low])
        high = np.concatenate([np.array([1000.0]), self.observation_space.high])

        self.observation_space = gym.spaces.Box(low, high, shape=(5,), dtype=np.float32)

        self.timesteps_ = 0

    def reset(self, *, seed=None, options=None):
        ret = super().reset()
        self.timesteps_ = 0
        obs = np.concatenate([np.array([self.timesteps_]), ret[0]])
        return obs, ret[1]

    def step(self, action):
        ret = super().step(action)

        self.timesteps_ += 1

        obs = np.concatenate([np.array([self.timesteps_]), ret[0]])
        reward = 0.1 * self.timesteps_
        return (obs, reward) + ret[2:]


gym.register("CartPoleDebug-v0", CartPoleDebug)
env = gym.make("CartPoleDebug-v0", render_mode="rgb_array")
env.reset()


def create_cartpole_dream_image(
    dreamed_obs,  # real space (not symlog'd)
    dreamed_V,  # real space (not symlog'd)
    dreamed_a,
    dreamed_r,  # real space (not symlog'd)
    dreamed_c,  # continue flag
    value_target,  # real space (not symlog'd)
    as_tensor=False,
):
    # Set the state of our env to the given observation.
    env.unwrapped.state = np.array(dreamed_obs[1:], dtype=np.float32)
    # Produce an RGB-image of the current state.
    rgb_array = env.render()

    # Add value-, action-, reward-, and continue-prediction information.
    image = Image.fromarray(rgb_array)
    draw_obj = ImageDraw.Draw(image)
    draw_obj.text((20, 26), f"V={dreamed_V} (target={value_target})", fill=(0, 0, 0))
    draw_obj.text((20, 38), f"a={'<--' if dreamed_a == 0 else '-->'} ({dreamed_a})", fill=(0, 0, 0))
    draw_obj.text((20, 50), f"r={dreamed_r}", fill=(0, 0, 0))
    draw_obj.text((20, 62), f"cont={dreamed_c}", fill=(0, 0, 0))

    draw_obj.text((20, 100), f"t={dreamed_obs[0]}", fill=(0, 0, 0))

    # Return image.
    np_img = np.asarray(image)
    if as_tensor:
        return tf.convert_to_tensor(np_img, dtype=tf.uint8)
    return np_img


if __name__ == "__main__":
    rgb_array = create_cartpole_dream_image(
        dreamed_obs=np.array([100.0, 1.0, -0.01, 1.5, 0.02]),
        dreamed_V=4.3,
        dreamed_a=1,
        dreamed_r=1.0,
        dreamed_c=True,
        value_target=8.0,
    )
    #ImageFont.load("arial.pil")
    image = Image.fromarray(rgb_array)
    image.show()
