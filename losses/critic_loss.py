import numpy as np
import scipy
import tensorflow as tf


@tf.function
def critic_loss(
        dreamed_observations,
        dreamed_rewards,
        terminateds,
        truncateds,
        forward_train_outs,
):
    pass # TODO


def gae(rewards, value_predictions, last_r, gamma, lambda_):
    value_predictions = np.concatenate([value_predictions, np.array([last_r])])
    delta_t = rewards + gamma * value_predictions[1:] - value_predictions[:-1]
    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    advantages = discounted_return_to_go(delta_t, gamma * lambda_)
    targets = (advantages + value_predictions[:-1]).astype(np.float32)
    return targets


def my_gae(rewards, continues, value_predictions, last_r, gamma, lambda_):
    value_predictions = np.concatenate([value_predictions, np.array([last_r])])
    Rs = [last_r]
    for i in reversed(range(len(rewards))):
        Rt = rewards[i] + gamma * continues[i] * ((1.0 - lambda_) * value_predictions[i+1] + lambda_ * Rs[-1])
        Rs.append(Rt)
    return np.flip(np.stack(Rs[1:]), axis=0)


def discounted_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted returns to go over a reward sequence.

    Moving from the end backwards in time until the beginning of the reward
    sequence, at each step, we compute:
    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.

    Examples:
        >>> rewards = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9**2*2.0 + 0.9**3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9**2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], rewards[::-1], axis=0)[::-1]


if __name__ == "__main__":
    expected = np.array([
        0.0 + 0.9 * 1.0 + 0.9 ** 2 * 2.0 + 0.9 ** 3 * 3.0,
        1.0 + 0.9 * 2.0 + 0.9 ** 2 * 3.0,
        2.0 + 0.9 * 3.0,
        3.0,
    ])
    print("expected:", expected)
    print("actual:")
    print(discounted_return_to_go(
        np.array([0.0, 1.0, 2.0, 3.0]),
        0.9,
    ))

    r = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    c = np.array([1.0, 0.0, 1.0, 1.0, 1.0])
    vf = np.array([2.0, 2.5, 3.1, -0.1, 0.3])
    last_r = 0.5
    gamma = 0.94
    lambda_ = 0.7

    # my GAE:
    print(my_gae(r, c, vf, last_r, gamma, lambda_))
    # RLlib GAE
    print(gae(r, vf, last_r, gamma, lambda_))
