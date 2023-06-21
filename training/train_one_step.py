"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf

from losses.actor_loss import actor_loss
from losses.critic_loss import compute_value_targets, critic_loss
from losses.disagree_loss import disagree_loss
from losses.world_model_losses import (
    world_model_dynamics_and_representation_loss,
    world_model_prediction_losses,
)


@tf.function
def train_world_model_one_step(
    *,
    sample,
    batch_size_B,
    batch_length_T,
    grad_clip,
    world_model,
):
    # Compute losses.
    with tf.GradientTape() as tape:
        # Compute forward (train) data.
        forward_train_outs = world_model.forward_train(
            observations=sample["obs"],
            actions=sample["actions"],
            is_first=sample["is_first"],
        )

        prediction_losses = world_model_prediction_losses(
            rewards=sample["rewards"],
            continues=(1.0 - sample["is_terminated"]),
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )

        L_dyn_B_T, L_rep_B_T = world_model_dynamics_and_representation_loss(
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )

        L_pred_B_T = prediction_losses["prediction_loss_B_T"]
        L_pred = tf.reduce_mean(L_pred_B_T)

        L_decoder_B_T = prediction_losses["decoder_loss_B_T"]
        L_decoder = tf.reduce_mean(L_decoder_B_T)

        # Two-hot reward loss.
        L_reward_two_hot_B_T = prediction_losses["reward_loss_two_hot_B_T"]
        L_reward_two_hot = tf.reduce_mean(L_reward_two_hot_B_T)
        # TEST: Out of interest, compare with simplge -log(p) loss for individual
        # rewards using the FiniteDiscrete distribution. This should be very close
        # to the two-hot reward loss.
        # L_reward_logp_B_T = prediction_losses["reward_loss_logp_B_T"]
        # L_reward_logp = tf.reduce_mean(L_reward_logp_B_T)

        L_continue_B_T = prediction_losses["continue_loss_B_T"]
        L_continue = tf.reduce_mean(L_continue_B_T)

        L_dyn = tf.reduce_mean(L_dyn_B_T)

        L_rep = tf.reduce_mean(L_rep_B_T)

        # Make sure values for L_rep and L_dyn are the same (they only differ in their gradients).
        tf.assert_equal(L_dyn, L_rep)

        # Compute the actual total loss using fixed weights described in [1] eq. 4.
        L_world_model_total_B_T = 1.0 * L_pred_B_T + 0.5 * L_dyn_B_T + 0.1 * L_rep_B_T

        # Sum up timesteps, and average over batch (see eq. 4 in [1]).
        L_world_model_total = tf.reduce_mean(L_world_model_total_B_T)

    # Get the gradients from the tape.
    gradients = tape.gradient(L_world_model_total, world_model.trainable_variables)
    # Clip all gradients by global norm.
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, grad_clip)
    # Apply gradients to our model.
    world_model.optimizer.apply_gradients(
        zip(clipped_gradients, world_model.trainable_variables)
    )

    return {
        # Forward train results.
        "WORLD_MODEL_forward_train_outs": forward_train_outs,
        "WORLD_MODEL_learned_initial_h": world_model.initial_h,
        # Prediction losses.
        # Decoder (obs) loss.
        "WORLD_MODEL_L_decoder_B_T": L_decoder_B_T,
        "WORLD_MODEL_L_decoder": L_decoder,
        # Reward loss.
        "WORLD_MODEL_L_reward_B_T": L_reward_two_hot_B_T,
        "WORLD_MODEL_L_reward": L_reward_two_hot,
        # Continue loss.
        "WORLD_MODEL_L_continue_B_T": L_continue_B_T,
        "WORLD_MODEL_L_continue": L_continue,
        # Total.
        "WORLD_MODEL_L_prediction_B_T": L_pred_B_T,
        "WORLD_MODEL_L_prediction": L_pred,
        # Dynamics loss.
        "WORLD_MODEL_L_dynamics_B_T": L_dyn_B_T,
        "WORLD_MODEL_L_dynamics": L_dyn,
        # Representation loss.
        "WORLD_MODEL_L_representation_B_T": L_rep_B_T,
        "WORLD_MODEL_L_representation": L_rep,
        # Total loss.
        "WORLD_MODEL_L_total_B_T": L_world_model_total_B_T,
        "WORLD_MODEL_L_total": L_world_model_total,
        # Gradient stats.
        "WORLD_MODEL_gradients_maxabs": (
            tf.reduce_max([tf.reduce_max(tf.math.abs(g)) for g in gradients])
        ),
        "WORLD_MODEL_gradients_clipped_by_glob_norm_maxabs": (
            tf.reduce_max([tf.reduce_max(tf.math.abs(g)) for g in clipped_gradients])
        ),
    }


@tf.function
def train_actor_and_critic_one_step(
    *,
    world_model_forward_train_outs,
    is_terminated,
    horizon_H,
    gamma,
    lambda_,
    actor_grad_clip,
    critic_grad_clip,
    disagree_grad_clip,
    dreamer_model,
    entropy_scale,
    return_normalization_decay,
    train_actor=True,
    use_curiosity=False,
):
    # Compute losses.
    with tf.GradientTape(persistent=True) as tape:
        # Dream trajectories starting in all internal states (h+z) that were
        # computed during world model training.
        dream_data = dreamer_model.dream_trajectory(
            start_states={
                "h": world_model_forward_train_outs["h_states_BxT"],
                "z": world_model_forward_train_outs["z_states_BxT"],
            },
            start_is_terminated=is_terminated,
            timesteps_H=horizon_H,
            gamma=gamma,
        )

        value_targets = compute_value_targets(
            # Learn critic in symlog'd space.
            rewards_t0_to_H_BxT=dream_data["rewards_dreamed_t0_to_H_B"],
            intrinsic_rewards_t1_to_H_BxT=(
                dream_data["rewards_intrinsic_t1_to_H_B"] if use_curiosity else None
            ),
            continues_t0_to_H_BxT=dream_data["continues_dreamed_t0_to_H_B"],
            value_predictions_t0_to_H_BxT=dream_data["values_dreamed_t0_to_H_B"],
            gamma=gamma,
            lambda_=lambda_,
        )
        critic_loss_results = critic_loss(dream_data, value_targets)
        if train_actor:
            actor_loss_results = actor_loss(
                dream_data=dream_data,
                value_targets=value_targets,
                actor=dreamer_model.actor,
                entropy_scale=entropy_scale,
                return_normalization_decay=return_normalization_decay,
            )
        if use_curiosity:
            L_disagree = disagree_loss(dream_data=dream_data)

    results = critic_loss_results.copy()
    results["VALUE_TARGETS_H_B"] = value_targets

    results["dream_data"] = dream_data
    L_critic = results["CRITIC_L_total"]

    # Get the gradients from the tape.
    if train_actor:
        results.update(actor_loss_results)
        L_actor = results["ACTOR_L_total"]
        actor_gradients = tape.gradient(
            L_actor,
            dreamer_model.actor.trainable_variables,
        )
    critic_gradients = tape.gradient(
        L_critic,
        dreamer_model.critic.trainable_variables,
    )
    if use_curiosity:
        results["DISAGREE_L_total"] = L_disagree
        results["DISAGREE_intrinsic_rewards_H_B"] = dream_data[
            "rewards_intrinsic_t1_to_H_B"
        ]
        results["DISAGREE_intrinsic_rewards"] = tf.reduce_mean(
            dream_data["rewards_intrinsic_t1_to_H_B"]
        )
        disagree_gradients = tape.gradient(
            L_disagree,
            dreamer_model.disagree_nets.trainable_variables,
        )

    # Clip all gradients by global norm.
    if train_actor:
        clipped_actor_gradients, _ = tf.clip_by_global_norm(
            actor_gradients, actor_grad_clip
        )
        results["ACTOR_gradients_maxabs"] = tf.reduce_max(
            [tf.reduce_max(tf.math.abs(g)) for g in actor_gradients]
        )
        results["ACTOR_gradients_clipped_by_glob_norm_maxabs"] = tf.reduce_max(
            [tf.reduce_max(tf.math.abs(g)) for g in clipped_actor_gradients]
        )
    clipped_critic_gradients, _ = tf.clip_by_global_norm(
        critic_gradients, critic_grad_clip
    )
    results["CRITIC_gradients_maxabs"] = tf.reduce_max(
        [tf.reduce_max(tf.math.abs(g)) for g in critic_gradients]
    )
    results["CRITIC_gradients_clipped_by_glob_norm_maxabs"] = tf.reduce_max(
        [tf.reduce_max(tf.math.abs(g)) for g in clipped_critic_gradients]
    )
    if use_curiosity:
        clipped_disagree_gradients, _ = tf.clip_by_global_norm(
            disagree_gradients, disagree_grad_clip
        )
        results["DISAGREE_gradients_maxabs"] = tf.reduce_max(
            [tf.reduce_max(tf.math.abs(g)) for g in disagree_gradients]
        )
        results["DISAGREE_gradients_clipped_by_glob_norm_maxabs"] = tf.reduce_max(
            [tf.reduce_max(tf.math.abs(g)) for g in clipped_disagree_gradients]
        )

    # Apply gradients to our models.
    if train_actor:
        dreamer_model.actor.optimizer.apply_gradients(
            zip(clipped_actor_gradients, dreamer_model.actor.trainable_variables)
        )
    dreamer_model.critic.optimizer.apply_gradients(
        zip(clipped_critic_gradients, dreamer_model.critic.trainable_variables)
    )
    if use_curiosity:
        dreamer_model.disagree_nets.optimizer.apply_gradients(
            zip(
                clipped_disagree_gradients,
                dreamer_model.disagree_nets.trainable_variables,
            )
        )

    # Update EMA weights of the critic.
    dreamer_model.critic.update_ema()

    return results
