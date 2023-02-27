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
    value_targets,
    actor,
    entropy_scale: float = 3e-4,
    return_normalization_decay: float = 0.99,
):
    scaled_value_targets_B_H = compute_scaled_value_targets(
        value_targets=value_targets,
        value_predictions=dream_data["values_dreamed_t1_to_Hp1"][:, :-1],
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
        actions_dreamed_one_hot * tf.math.log(probs_actions_B_H),
        axis=-1,
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

    L_actor_reinforce_term_B_H = - logp_loss_B_H
    L_actor_action_entropy_term_B_H = - entropy_scale * entropy_B_H

    L_actor_B_H = L_actor_reinforce_term_B_H + L_actor_action_entropy_term_B_H
    L_actor = tf.reduce_mean(tf.reduce_sum(L_actor_B_H, axis=-1))

    return {
        "L_actor_B_H": L_actor_B_H,
        "L_actor": L_actor,
        "logp_actions_dreamed_B_H": logp_actions_dreamed_B_H,
        "scaled_value_targets_B_H": scaled_value_targets_B_H,
        "L_actor_ema_value_target_pct95": actor.ema_value_target_pct95,
        "L_actor_ema_value_target_pct5": actor.ema_value_target_pct5,
        "logp_loss_B_H": logp_loss_B_H,
        "action_entropy_B_H": entropy_B_H,
        "action_entropy": entropy,
        "L_actor_reinforce_term_B_H": L_actor_reinforce_term_B_H,
        "L_actor_reinforce_term": tf.reduce_mean(
            tf.reduce_sum(L_actor_reinforce_term_B_H, axis=-1)
        ),
        "L_actor_action_entropy_term_B_H": L_actor_action_entropy_term_B_H,
        "L_actor_action_entropy_term": tf.reduce_mean(
            tf.reduce_sum(L_actor_action_entropy_term_B_H, axis=-1)
        ),
    }


def compute_scaled_value_targets(
    *,
    value_targets,
    value_predictions,
    actor,
    return_normalization_decay: float = 0.99,
):
    value_targets_B_H = value_targets
    value_predictions_B_H = value_predictions

    # Compute S: [1] eq. 12.
    Per_R_5 = tfp.stats.percentile(value_targets_B_H, 5)
    Per_R_95 = tfp.stats.percentile(value_targets_B_H, 95)

    # Update EMAs stored in actor network.
    # Initial values: Just set.
    if tf.math.is_nan(actor.ema_value_target_pct5):
        actor.ema_value_target_pct5.assign(Per_R_5)
        actor.ema_value_target_pct95.assign(Per_R_95)
    # Later update (something already stored in EMA variable): Update EMA.
    else:
        actor.ema_value_target_pct5.assign(
            return_normalization_decay * actor.ema_value_target_pct5 + (
                1.0 - return_normalization_decay
            ) * Per_R_5
        )
        actor.ema_value_target_pct95.assign(
            return_normalization_decay * actor.ema_value_target_pct95 + (
                1.0 - return_normalization_decay
            ) * Per_R_95
        )

    # [1] eq. 11 (first term).
    #scaled_value_targets_B_H = tf.stop_gradient(
    #    value_targets_B_H / tf.math.maximum(
    #        1.0,
    #        actor.ema_value_target_pct95 - actor.ema_value_target_pct5,
    #   )
    #)
    # Dani's code: TODO: describe ...
    offset = actor.ema_value_target_pct5
    invscale = tf.math.maximum(1e-8, (actor.ema_value_target_pct95 - actor.ema_value_target_pct5))
    scaled_value_targets_B_H = (value_targets_B_H - offset) / invscale
    scaled_value_predictions_B_H = (value_predictions_B_H - offset) / invscale
    # Return advantages.
    return tf.stop_gradient(scaled_value_targets_B_H - scaled_value_predictions_B_H)
