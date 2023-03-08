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
    H, B = dream_data["rewards_dreamed_t0_to_H_B"].shape[:2]
    Hm1 = H - 1
    # Note that value targets are NOT symlog'd and start at t1, not t0, like all the
    # other dream data.
    value_targets_t0_to_Hm1_B = compute_value_targets(
        # Learn critic in symlog'd space.
        rewards_H_BxT=dream_data["rewards_dreamed_t0_to_H_B"],
        continues_H_BxT=dream_data["continues_dreamed_t0_to_H_B"],
        value_predictions_H_BxT=dream_data["values_dreamed_t0_to_H_B"],
        gamma=gamma,
        lambda_=lambda_,
    )
    value_symlog_targets_t0_to_Hm1_B = symlog(value_targets_t0_to_Hm1_B)
    # Fold time rank (for two_hot'ing).
    value_symlog_targets_HxB = tf.reshape(value_symlog_targets_t0_to_Hm1_B, (-1,))
    value_symlog_targets_two_hot_HxB = two_hot(value_symlog_targets_HxB)
    # Unfold time rank.
    value_symlog_targets_two_hot_t0_to_Hm1_B = tf.reshape(
        value_symlog_targets_two_hot_HxB,
        shape=[Hm1, B, value_symlog_targets_two_hot_HxB.shape[-1]],
    )
    # Get (B x T x probs) tensor from return distributions.
    value_symlog_logits_HxB = dream_data["values_symlog_dreamed_logits_t0_to_HxB"]
    # Unfold time rank and cut last time index to match value targets.
    value_symlog_logits_t0_to_Hm1_B = tf.reshape(
        value_symlog_logits_HxB,
        shape=[H, B, value_symlog_logits_HxB.shape[-1]],
    )[:-1]

    values_log_pred_Hm1_B = value_symlog_logits_t0_to_Hm1_B - tf.math.reduce_logsumexp(value_symlog_logits_t0_to_Hm1_B, axis=-1, keepdims=True)
    # Multiply with two-hot targets and neg.
    value_loss_two_hot_H_B = - tf.reduce_sum(values_log_pred_Hm1_B * value_symlog_targets_two_hot_t0_to_Hm1_B, axis=-1)
    ## Unfold time rank back in.
    #value_symlog_neg_logp_target_t0_to_Hm1_B = tf.reshape(value_loss_two_hot_HxB, (Hm1, B))

    # Vector product to reduce over return-buckets. See [1] eq. 10.
    #value_symlog_logp_B_H = tf.reduce_sum(
    #    tf.multiply(
    #        tf.stop_gradient(value_symlog_targets_two_hot_B_H),
    #        tf.math.log(value_symlog_probs_B_H),
    #    ),
    #    axis=-1,
    #)
    #value_symlog_neg_logp_target_t0_to_Hm1_B = - tf.math.log(tf.reduce_sum(
    #    tf.multiply(
    #        tf.stop_gradient(value_symlog_targets_two_hot_t0_to_Hm1_B),
    #        value_symlog_probs_t0_to_Hm1_B,
    #    ),
    #    axis=-1,
    #))
    #value_symlog_neg_logp_target_H_B = tf.reshape(
    #    value_symlog_neg_logp_target_HxB,
    #    shape=[H, B],# + value_symlog_probs_HxB.shape[-1],
    #)

    # Compute EMA regularization loss.
    # Expected values (dreamed) from the EMA (slow critic) net.
    value_symlog_ema_t0_to_Hm1_B = dream_data["v_symlog_dreamed_ema_t0_to_H_B"][:-1]
    # Fold time rank (for two_hot'ing).
    value_symlog_ema_HxB = tf.reshape(value_symlog_ema_t0_to_Hm1_B, (-1,))
    value_symlog_ema_two_hot_HxB = two_hot(value_symlog_ema_HxB)
    # Unfold time rank.
    value_symlog_ema_two_hot_t0_to_Hm1_B = tf.reshape(
        value_symlog_ema_two_hot_HxB,
        shape=[Hm1, B, value_symlog_ema_two_hot_HxB.shape[-1]],
    )

    # Compute ema regularizer loss.
    # In the paper, it is not described how exactly to form this regularizer term and
    # how to weigh it.
    # So we follow Dani's repo here: `reg = -dist.log_prob(sg(self.slow(traj).mean()))`
    # with a weight of 1.0, where dist is the bucket'ized distribution output by the
    # fast critic. sg=stop gradient; mean() -> use the expected EMA values.
    #values_log_pred_HxB = value_symlog_logits_t0_to_Hm1_B - tf.math.reduce_logsumexp(value_symlog_logits_t0_to_Hm1_B, axis=-1, keepdims=True)
    # Multiply with two-hot targets and neg.
    ema_regularization_loss_H_B = - tf.reduce_sum(values_log_pred_Hm1_B * value_symlog_ema_two_hot_t0_to_Hm1_B, axis=-1)
    ## Unfold time rank back in.
    #ema_regularization_loss_t0_to_Hm1_B = tf.reshape(ema_regularization_loss_HxB, (Hm1, B))

    #ema_regularization_loss_t0_to_Hm1_B = - tf.math.log(tf.reduce_sum(
    #    tf.multiply(
    #        tf.stop_gradient(value_symlog_ema_two_hot_t0_to_Hm1_B),
    #        value_symlog_probs_t0_to_Hm1_B,
    #    ),
    #    axis=-1,
    #))
    # Unfold time rank.
    #ema_regularization_loss_H_B = tf.reshape(
    #    ema_regularization_loss_HxB,
    #    shape=[Hm1, B],# + ema_regularization_loss_HxB.shape[-1],
    #)

    # Using MSE on the outputs (probs) of the EMA net vs the fast critic.
    #ema_regularization_loss_B_H = 0.5 * tf.reduce_sum(
    #    tf.math.square(
    #        value_symlog_probs_B_H - tf.stop_gradient(value_symlog_ema_probs_B_H)
    #    ),
    #    axis=-1,
    #)
    #L_critic_neg_logp_target = tf.reduce_mean(value_symlog_neg_logp_target_HxB)
    #L_critic_ema_regularization = tf.reduce_mean(ema_regularization_loss_B_H)

    L_critic_H_B = (
        value_loss_two_hot_H_B + ema_regularization_loss_H_B
    )

    # Mask out everything that goes beyond a predicted continue=False boundary.
    L_critic_H_B *= dream_data["dream_loss_weights_t0_to_H_B"][:-1]

    # Reduce over H- (time) axis (sum) and then B-axis (mean).
    L_critic = tf.reduce_mean(L_critic_H_B)

    return {
        "L_critic": L_critic,
        "L_critic_H_B": L_critic_H_B,
        "value_targets_H_B": value_targets_t0_to_Hm1_B,
        "value_symlog_targets_H_B": value_symlog_targets_t0_to_Hm1_B,
        #"value_probs_B_H": value_probs_B_H,
        #"L_critic_neg_logp_target": L_critic_neg_logp_target,
        "L_critic_neg_logp_target_H_B": value_loss_two_hot_H_B,
        #"L_critic_ema_regularization": L_critic_ema_regularization,
        "L_critic_ema_regularization_H_B": ema_regularization_loss_H_B,
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

    Rewards, continues, and value_predictions are all of shape [t0-H, B] (time-major),
    whereas returned targets are [t0 to H-1, B] (last timestep missing as target equals
    vf prediction in that location.
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
    # targets.shape=[t0 to H-1,BxT]

    # Value targets should not be backpropped through.
    # They are used for critic learning (making the value function predict
    # these targets over time) and actor learning (for computing the "scaled value
    # targets", which are used for the advanatges with which we weigh the actions).
    return tf.stop_gradient(targets)

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
