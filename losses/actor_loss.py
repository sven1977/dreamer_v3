"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf
import tensorflow_probability as tfp


@tf.function
def actor_loss(
    *,
    dream_data,
    critic_loss_results,
    actor,
    entropy_scale: float = 3e-4,
    return_normalization_decay: float = 0.99,
):
    scaled_value_targets_B_H = compute_scaled_value_targets(
        critic_loss_results=critic_loss_results,
        actor=actor,
        return_normalization_decay=return_normalization_decay,
    )

    # Actions actually taken in the dream.
    actions_dreamed = dream_data["actions_dreamed_t1_to_H"]
    # Probabilities of all possible actions in the dream.
    probs_actions_B_H = tf.stack(
        [dist.probs for dist in dream_data["actions_dreamed_distributions_t1_to_H"]],
        axis=1,
    )
    num_actions = probs_actions_B_H.shape[-1]
    actions_dreamed_one_hot = tf.one_hot(actions_dreamed, depth=num_actions)
    # Log probs of actions actually taken in the dream.
    logp_actions_dreamed_B_H = tf.reduce_sum(
        actions_dreamed_one_hot * tf.math.log(probs_actions_B_H), axis=-1
    )
    # First term of loss function.
    # [1] eq. 11.
    logp_loss_B_H = logp_actions_dreamed_B_H * scaled_value_targets_B_H
    assert len(logp_loss_B_H.shape) == 2
    # Add entropy loss term (second term [1] eq. 11).
    entropy_B_H = tf.stack(
        [dist.entropy() for dist in dream_data["actions_dreamed_distributions_t1_to_H"]],
        axis=1,
    )
    assert len(entropy_B_H.shape) == 2
    entropy = tf.reduce_mean(tf.reduce_sum(entropy_B_H, axis=-1))

    L_actor_B_H = - logp_loss_B_H - entropy_scale * entropy_B_H
    L_actor = tf.reduce_mean(tf.reduce_sum(L_actor_B_H, axis=-1))

    return {
        "L_actor_B_H": L_actor_B_H,
        "L_actor": L_actor,
        "logp_actions_dreamed_B_H": logp_actions_dreamed_B_H,
        "scaled_value_targets_B_H": scaled_value_targets_B_H,
        "logp_loss_B_H": logp_loss_B_H,
        "action_entropy_B_H": entropy_B_H,
        "action_entropy": entropy,
    }


def compute_scaled_value_targets(
    *,
    critic_loss_results,
    actor,
    return_normalization_decay: float = 0.99,
):
    value_targets_B_H = critic_loss_results["value_targets_B_H"]

    # Compute S: [1] eq. 12.
    Per_R_95 = tfp.stats.percentile(value_targets_B_H, 95)
    Per_R_5 = tfp.stats.percentile(value_targets_B_H, 5)
    Per_R_95_m_5 = Per_R_95 - Per_R_5

    # Update EMA stored in actor network.
    # Initial value.
    if tf.math.is_nan(actor.ema_range_95_minus_5):
        actor.ema_range_95_minus_5.assign(Per_R_95_m_5)
    # Later update (something already stored in EMA variable).
    else:
        actor.ema_range_95_minus_5.assign(
            return_normalization_decay * actor.ema_range_95_minus_5 + (
                1.0 - return_normalization_decay
            ) * Per_R_95_m_5
        )

    # [1] eq. 11 (first term).
    scaled_value_targets_B_H = tf.stop_gradient(
        value_targets_B_H / tf.math.maximum(
            1.0,
            actor.ema_range_95_minus_5,
       )
    )
    return scaled_value_targets_B_H
