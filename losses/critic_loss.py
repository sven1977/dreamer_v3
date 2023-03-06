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
    # Note that value targets are NOT symlog'd.
    value_targets_B_H = compute_value_targets(
        # Learn critic in symlog'd space.
        rewards=dream_data["rewards_dreamed_t1_to_Hp1"],
        continues=dream_data["continues_dreamed_t1_to_Hp1"],
        value_predictions=dream_data["values_dreamed_t1_to_Hp1"],
        gamma=gamma,
        lambda_=lambda_,
    )
    value_symlog_targets_B_H = symlog(value_targets_B_H)
    # Fold time rank (for two_hot'ing).
    value_symlog_targets_BxH = tf.reshape(value_symlog_targets_B_H, (-1,))
    value_symlog_targets_BxH_two_hot = two_hot(value_symlog_targets_BxH)
    # Unfold time rank.
    value_symlog_targets_two_hot_B_H = tf.reshape(
        value_symlog_targets_BxH_two_hot,
        shape=value_symlog_targets_B_H.shape[:2] + value_symlog_targets_BxH_two_hot.shape[-1],
    )
    # Get (B x T x probs) tensor from return distributions.
    value_symlog_probs_B_H = tf.stack(
        [d.probs for d in dream_data["values_symlog_dreamed_distributions_t1_to_Hp1"][:-1]],
        axis=1,
    )
    # Vector product to reduce over return-buckets. See [1] eq. 10.
    #value_symlog_logp_B_H = tf.reduce_sum(
    #    tf.multiply(
    #        tf.stop_gradient(value_symlog_targets_two_hot_B_H),
    #        tf.math.log(value_symlog_probs_B_H),
    #    ),
    #    axis=-1,
    #)
    value_symlog_neg_logp_target_B_H = - tf.math.log(tf.reduce_sum(
        tf.multiply(
            tf.stop_gradient(value_symlog_targets_two_hot_B_H),
            value_symlog_probs_B_H,
        ),
        axis=-1,
    ))

    # Compute EMA regularization loss.
    # Expected values (dreamed) from the EMA (slow critic) net.
    value_symlog_ema_B_H = dream_data["v_symlog_dreamed_ema_t1_to_Hp1"][:, :-1]
    # Fold time rank (for two_hot'ing).
    value_symlog_ema_BxH = tf.reshape(value_symlog_ema_B_H, (-1,))
    value_symlog_ema_two_hot_BxH = two_hot(value_symlog_ema_BxH)
    # Unfold time rank.
    value_symlog_ema_two_hot_B_H = tf.reshape(
        value_symlog_ema_two_hot_BxH,
        shape=value_symlog_targets_B_H.shape[:2] + value_symlog_ema_two_hot_BxH.shape[-1],
    )
    # Compute ema regularizer loss.
    # In the paper, it is not described how exactly to form this regularizer term and
    # how to weigh it.
    # So we follow Dani's repo here: `reg = -dist.log_prob(sg(self.slow(traj).mean()))`
    # with a weight of 1.0, where dist is the bucket'ized distribution output by the
    # fast critic. sg=stop gradient; mean() -> use the expected EMA values.
    ema_regularization_loss_B_H = - tf.math.log(tf.reduce_sum(
        tf.multiply(
            tf.stop_gradient(value_symlog_ema_two_hot_B_H),
            value_symlog_probs_B_H,
        ),
        axis=-1,
    ))
    # Using MSE on the outputs (probs) of the EMA net vs the fast critic.
    #ema_regularization_loss_B_H = 0.5 * tf.reduce_sum(
    #    tf.math.square(
    #        value_symlog_probs_B_H - tf.stop_gradient(value_symlog_ema_probs_B_H)
    #    ),
    #    axis=-1,
    #)
    L_critic_neg_logp_target = tf.reduce_mean(value_symlog_neg_logp_target_B_H)
    L_critic_ema_regularization = tf.reduce_mean(ema_regularization_loss_B_H)

    L_critic_B_H = value_symlog_neg_logp_target_B_H + ema_regularization_loss_B_H

    # Reduce over H- (time) axis (sum) and then B-axis (mean).
    L_critic = tf.reduce_mean(L_critic_B_H)

    return {
        "L_critic": L_critic,
        "L_critic_B_H": L_critic_B_H,
        "value_targets_B_H": value_targets_B_H,
        "value_symlog_targets_B_H": value_symlog_targets_B_H,
        #"value_probs_B_H": value_probs_B_H,
        "L_critic_neg_logp_target": L_critic_neg_logp_target,
        "L_critic_neg_logp_target_B_H": value_symlog_neg_logp_target_B_H,
        "L_critic_ema_regularization": L_critic_ema_regularization,
        "L_critic_ema_regularization_B_H": ema_regularization_loss_B_H,
    }


def compute_value_targets(
    *,
    rewards_H_BxT,
    continues_H_BxT,
    value_predictions_H_BxT,
    gamma,
    lambda_,
):
    """All args are (H, BxT, ...) and in non-symlog'd (real) space.

    Where H=1+horizon (start state + H steps dreamed), BxT=batch_size * batch_length
    (original trajectory time rank has been folded).

    Non-symlog is important b/c log(a+b) != log(a) + log(b).
    See [1] eq. 8 and 10.

    Thus, targets are always returned in real (non-symlog'd space).
    They need to be re-symlog'd before computing the critic loss from them (b/c the
    critic does produce predictions in symlog space).

    Furthermore, rewards, continues, and value_predictions are all of shape [B, t1-H+1]
    """
    # In all the following, when building value targets for t=1 to T=H,
    # exclude rewards & continues for t=1 b/c we don't need r1 or c1.
    # The target (R1) for V1 is built from r2, c2, and V2/R2.
    discount = continues_H_BxT[1:] * gamma  # shape=[2-16, BxT]
    Rs = [value_predictions_H_BxT[-1]]  # Rs indices=[16]
    intermediates = rewards_H_BxT[1:] + discount * (1 - lambda_) * value_predictions_H_BxT[1:]
    # intermediates.shape=[2-16, BxT]

    # Loop through reversed timesteps (axis=1) from T+1 to t=2.
    for t in reversed(range(len(discount))):
        Rs.append(intermediates[t] + discount[t] * lambda_ * Rs[-1])

    # Reverse along time axis and cut the last entry (value estimate at very end cannot
    # be learnt from as it's the same as the ... well ... value estimate).
    targets = tf.stack(list(reversed(Rs))[:-1], axis=0)
    # targets.shape=[1-15,BxT]

    return targets

    # Danijar's code:
    # Note: All shapes are time-major: H=16, B=1024(==BxT), ...
    # rew = self.rewfn(traj)  # shape=[2-16, B(1024)]  # 2-16 means: includes 16, but excludes first reward (from initial porterior state)
    # discount = 1 - 1 / self.config.horizon
    # disc = traj['cont'][1:] * discount  # shape=[2-16, B]
    # value = self.net(traj).mean()  # shape=[1-16, B]
    # vals = [value[-1]]  # val indices = [16]
    # interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
    # interm.shape==[2-16, B]
    # for t in reversed(range(len(disc))):
    #   vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
    #   # val indices = [16, 15, 14, 13, 12, ..., 1]
    # ret = jnp.stack(list(reversed(vals))[:-1])
    # ret.shape=[1-15, B]  # value targets for values, except last (doesn't make sense as target == prediction)
    # return rew (1-15, B), ret (1-15, B), value[:-1] (1-15, B)


if __name__ == "__main__":
    r = np.array(
        [[99.0],  [1.0],  [2.0],  [3.0],  [4.0],  [5.0]],
        #[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    c = np.array(
        [ [1.0],  [1.0],  [0.0],  [1.0],  [1.0],  [1.0]],
        #[1.0,  0.0, 1.0, 1.0, 1.0],
    )
    vf = np.array(
        [ [3.0],  [2.0], [15.0], [12.0],  [8.0],  [3.0]],  # naive sum of future rewards
        #[7.0, 6.0, 5.0, 4.0, 3.0, 2.0],  # naive sum of future rewards
    )
    #last_r = vf[:, -1]
    gamma = 1.0#0.99
    lambda_ = 1.0#0.7

    # my GAE:
    print(compute_value_targets(
        rewards_H_BxT=r,
        continues_H_BxT=c,
        value_predictions_H_BxT=vf,
        gamma=gamma,
        lambda_=lambda_,
    ))
    # RLlib GAE
    #print(_rllib_gae(r, vf[:, -1], last_r, gamma, lambda_))
