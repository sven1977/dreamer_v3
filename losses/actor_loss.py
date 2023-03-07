"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf
import tensorflow_probability as tfp
from gymnasium.spaces import Discrete, Box


@tf.function
def actor_loss(
    *,
    dream_data,
    value_targets,
    actor,
    entropy_scale,
    return_normalization_decay,
):
    scaled_value_targets_t0_to_Hm1_B = compute_scaled_value_targets(
        value_targets=value_targets,  # targets are already [t0 to H-1, B]
        value_predictions=dream_data["values_dreamed_t0_to_H_B"][:-1],
        actor=actor,
        return_normalization_decay=return_normalization_decay,
    )

    # Actions actually taken in the dream.
    actions_dreamed = dream_data["actions_dreamed_t0_to_H_B"][:-1]
    # Log(p)s of all possible actions in the dream.
    # Note that when we create the Categorical action distributions, we compute
    # unimix probs, then math.log these and provide these log(p) as "logits" to the
    # Categorical. So here, we'll continue to work with log(p)s (not really "logits")!
    logp_actions_t0_to_Hm1_B = tf.stack(
        [dist.logits
         for dist in dream_data["actions_dreamed_distributions_t0_to_H_B"][:-1]
         ],
        axis=0,
    )
    num_actions = logp_actions_t0_to_Hm1_B.shape[-1]
    actions_dreamed_one_hot = tf.one_hot(actions_dreamed, depth=num_actions)
    # Log probs of actions actually taken in the dream.
    logp_actions_dreamed_t0_to_Hm1_B = tf.reduce_sum(
        actions_dreamed_one_hot * logp_actions_t0_to_Hm1_B,
        axis=-1,
    )
    # First term of loss function.
    # [1] eq. 11.
    if isinstance(actor.action_space, Discrete):
        logp_loss_H_B = logp_actions_dreamed_t0_to_Hm1_B * tf.stop_gradient(scaled_value_targets_t0_to_Hm1_B)
    elif isinstance(actor.action_space, Box):
        logp_loss_H_B = scaled_value_targets_t0_to_Hm1_B
    else:
        raise ValueError(f"Invalid action space: {actor.action_space}")
    assert len(logp_loss_H_B.shape) == 2
    # Add entropy loss term (second term [1] eq. 11).
    entropy_H_B = tf.stack(
        [dist.entropy()
         for dist in dream_data["actions_dreamed_distributions_t0_to_H_B"][:-1]
         ],
        axis=0,
    )
    assert len(entropy_H_B.shape) == 2
    entropy = tf.reduce_mean(entropy_H_B)

    L_actor_reinforce_term_H_B = - logp_loss_H_B
    L_actor_action_entropy_term_H_B = - entropy_scale * entropy_H_B

    L_actor_H_B = L_actor_reinforce_term_H_B + L_actor_action_entropy_term_H_B
    # Mask out everything that goes beyond a predicted continue=False boundary.
    L_actor_H_B *= dream_data["dream_loss_weights_t0_to_H_B"][:-1]
    L_actor = tf.reduce_mean(L_actor_H_B)

    return {
        "L_actor_H_B": L_actor_H_B,
        "L_actor": L_actor,
        "logp_actions_dreamed_H_B": logp_actions_dreamed_t0_to_Hm1_B,
        "scaled_value_targets_H_B": scaled_value_targets_t0_to_Hm1_B,
        "L_actor_ema_value_target_pct95": actor.ema_value_target_pct95,
        "L_actor_ema_value_target_pct5": actor.ema_value_target_pct5,
        "logp_loss_H_B": logp_loss_H_B,
        "action_entropy_B_H": entropy_H_B,
        "action_entropy": entropy,

        #"L_actor_reinforce_term_B_H": L_actor_reinforce_term_B_H,
        #"L_actor_reinforce_term": tf.reduce_mean(L_actor_reinforce_term_B_H),
        #"L_actor_action_entropy_term_B_H": L_actor_action_entropy_term_B_H,
        #"L_actor_action_entropy_term": tf.reduce_mean(L_actor_action_entropy_term_B_H),
    }


def compute_scaled_value_targets(
    *,
    value_targets,
    value_predictions,
    actor,
    return_normalization_decay,
):
    value_targets_H_B = value_targets
    value_predictions_H_B = value_predictions

    # Compute S: [1] eq. 12.
    Per_R_5 = tfp.stats.percentile(value_targets_H_B, 5)
    Per_R_95 = tfp.stats.percentile(value_targets_H_B, 95)

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
    scaled_value_targets_H_B = (value_targets_H_B - offset) / invscale
    scaled_value_predictions_H_B = (value_predictions_H_B - offset) / invscale
    # Return advantages.
    return tf.stop_gradient(scaled_value_targets_H_B - scaled_value_predictions_H_B)
