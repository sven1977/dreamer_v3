"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import scipy
import tensorflow as tf

from utils.symlog import symlog
from utils.two_hot import two_hot


@tf.function
def critic_loss(
        dream_data,
        gamma,
        lambda_,
):
    value_targets_B_H = compute_value_targets(
        # Learn critic in symlog'd space.
        rewards=dream_data["rewards_dreamed_t1_to_H"],
        continues=dream_data["continues_dreamed_t1_to_H"],
        value_predictions=dream_data["values_dreamed_t1_to_Hp1"],
        gamma=gamma,
        lambda_=lambda_,
    )
    value_symlog_targets_B_H = symlog(value_targets_B_H)
    # value_targets=(B, T)
    value_symlog_targets_BxH = tf.reshape(value_symlog_targets_B_H, (-1,))
    value_symlog_targets_BxH_two_hot = two_hot(value_symlog_targets_BxH)
    value_symlog_targets_two_hot_B_H = tf.reshape(
        value_symlog_targets_BxH_two_hot,
        shape=value_symlog_targets_B_H.shape[:2] + value_symlog_targets_BxH_two_hot.shape[-1],
    )
    # Get (B x T x probs) tensor from return distributions.
    value_symlog_probs_B_H = tf.stack(
        [d.probs for d in dream_data["values_symlog_dreamed_distributions_t1_to_Hp1"][:-1]],
        axis=1,
    )
    # Vector product to reduce over return-buckets. [1] eq. 10.
    value_symlog_logp_B_H = tf.reduce_sum(
        tf.multiply(
            tf.stop_gradient(value_symlog_targets_two_hot_B_H),
            tf.math.log(value_symlog_probs_B_H),
        ),
        axis=-1,
    )

    # Compute EMA L2-regularization loss.
    value_symlog_ema_probs_B_H = tf.stack(
        [d.probs for d in dream_data["v_symlog_dreamed_distributions_ema_t1_to_Hp1"][:-1]],
        axis=1,
    )
    ema_regularization_loss_B_H = 0.5 * tf.reduce_sum(
        tf.math.square(
            value_symlog_probs_B_H - tf.stop_gradient(value_symlog_ema_probs_B_H)
        ),
        axis=-1,
    )

    L_critic_B_H = -value_symlog_logp_B_H + ema_regularization_loss_B_H

    # Reduce over H- (time) axis (sum) and then B-axis (mean).
    L_critic = tf.reduce_mean(tf.reduce_sum(L_critic_B_H, axis=-1))

    return {
        "L_critic": L_critic,
        "L_critic_B_H": L_critic_B_H,
        "value_targets_B_H": value_targets_B_H,
        "value_symlog_targets_B_H": value_symlog_targets_B_H,
        #"value_probs_B_H": value_probs_B_H,
        "value_symlog_logp_B_H": value_symlog_logp_B_H,
        "ema_regularization_loss_B_H": ema_regularization_loss_B_H,
    }


def compute_value_targets(rewards, continues, value_predictions, gamma, lambda_):
    """All args are (B, T, ...) and in non-symlog'd (real) space.

    The latter is important b/c log(a+b) != log(a) + log(b).
    See [1] eq. 8 and 10.

    Thus, targets are always returned in real (non-symlog'd space).
    They need to be re-symlog'd before computing the critic loss from them (b/c the
    critic does produce predictions in symlog space).
    """
    last_Rs = value_predictions[:, -1]
    Rs = []
    # Loop through reversed timesteps (axis=1).
    for i in reversed(range(rewards.shape[1])):
        Rt = rewards[:, i] + gamma * continues[:, i] * ((1.0 - lambda_) * value_predictions[:, i+1] + lambda_ * last_Rs)
        last_Rs = Rt
        Rs.append(Rt)
    # Reverse along time axis and cut the last entry (value estimate at very end cannot
    # be learnt from as it's the same as the ... well ... value estimate).
    return tf.reverse(tf.stack(Rs, axis=1), axis=[1])


def _rllib_gae(rewards, value_predictions, last_r, gamma, lambda_):
    value_predictions = np.concatenate([value_predictions, np.array([last_r])])
    delta_t = rewards + gamma * value_predictions[1:] - value_predictions[:-1]
    # This formula for the advantage comes from:
    # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
    advantages = _rllib_discounted_return_to_go(delta_t, gamma * lambda_)
    targets = (advantages + value_predictions[:-1]).astype(np.float32)
    return targets


def _rllib_discounted_return_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
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
    print(_rllib_discounted_return_to_go(
        np.array([0.0, 1.0, 2.0, 3.0]),
        0.9,
    ))

    r = np.array([
        [0.0, 1.0, 2.0, 3.0, 4.0],
        #[1.0, 1.0, 1.0, 1.0, 1.0],
    ])
    c = np.array([
        [1.0, 1.0, 1.0, 1.0, 1.0],
        #[1.0, 0.0, 1.0, 1.0, 1.0],
    ])
    vf = np.array([
        [13.0, 13.0, 12.0, 10.0, 7.0, 3.0],  # naive sum of future rewards
        #[7.0, 6.0, 5.0, 4.0, 3.0, 2.0],  # naive sum of future rewards
    ])
    last_r = vf[:, -1]
    gamma = 1.0#0.99
    lambda_ = 1.0#0.7

    # my GAE:
    print(compute_value_targets(r, c, vf, gamma, lambda_))
    # RLlib GAE
    #print(_rllib_gae(r, vf[:, -1], last_r, gamma, lambda_))
