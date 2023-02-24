"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf
from losses.actor_loss import actor_loss
from losses.critic_loss import critic_loss
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
    optimizer,
):
    # Compute losses.
    with tf.GradientTape() as tape:
        # Compute forward (train) data.
        forward_train_outs = world_model.forward_train(
            observations=sample["obs"],
            actions=sample["actions"],
            initial_h=sample["h_states"][:, 0],
        )

        prediction_losses = world_model_prediction_losses(
            observations=sample["obs"],
            rewards=sample["rewards"],
            continues=sample["continues"],
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )

        L_dyn_B_T, L_rep_B_T = world_model_dynamics_and_representation_loss(
            B=tf.convert_to_tensor(batch_size_B),
            T=tf.convert_to_tensor(batch_length_T),
            forward_train_outs=forward_train_outs,
        )

        L_pred_B_T = prediction_losses["total_loss_B_T"]
        L_pred = tf.reduce_mean(tf.reduce_sum(L_pred_B_T, axis=-1))

        L_decoder_B_T = prediction_losses["decoder_loss_B_T"]
        L_decoder = tf.reduce_mean(tf.reduce_sum(L_decoder_B_T, axis=-1))

        # Two-hot reward loss.
        L_reward_two_hot_B_T = prediction_losses["reward_loss_two_hot_B_T"]
        L_reward_two_hot = tf.reduce_mean(tf.reduce_sum(L_reward_two_hot_B_T, axis=-1))
        # TEST: Out of interest, compare with simplge -log(p) loss for individual
        # rewards using the FiniteDiscrete distribution. This should be very close
        # to the two-hot reward loss.
        L_reward_logp_B_T = prediction_losses["reward_loss_logp_B_T"]
        L_reward_logp = tf.reduce_mean(tf.reduce_sum(L_reward_logp_B_T, axis=-1))

        L_continue_B_T = prediction_losses["continue_loss_B_T"]
        L_continue = tf.reduce_mean(tf.reduce_sum(L_continue_B_T, axis=-1))

        L_dyn = tf.reduce_mean(tf.reduce_sum(L_dyn_B_T, axis=-1))

        L_rep = tf.reduce_mean(tf.reduce_sum(L_rep_B_T, axis=-1))

        # Compute the actual total loss using fixed weights described in [1] eq. 4.
        L_world_model_total_B_T = 1.0 * L_pred_B_T + 0.5 * L_dyn_B_T + 0.1 * L_rep_B_T

        # Sum up timesteps, and average over batch (see eq. 4 in [1]).
        L_world_model_total = tf.reduce_mean(tf.reduce_sum(L_world_model_total_B_T, axis=-1))

    # Get the gradients from the tape.
    gradients = tape.gradient(L_world_model_total, world_model.trainable_variables)
    # Clip all gradients.
    clipped_gradients = []
    for grad in gradients:
        clipped_gradients.append(tf.clip_by_value(grad, -grad_clip, grad_clip))
    # Apply gradients to our model.
    optimizer.apply_gradients(zip(clipped_gradients, world_model.trainable_variables))

    return {
        # Forward train results.
        "forward_train_outs": forward_train_outs,

        # Prediction losses.
        # Total.
        "L_pred_B_T": L_pred_B_T,
        "L_pred": L_pred,
        # Decoder.
        "L_decoder_B_T": L_decoder_B_T,
        "L_decoder": L_decoder,
        # Reward (two-hot).
        "L_reward_two_hot_B_T": L_reward_two_hot_B_T,
        "L_reward_two_hot": L_reward_two_hot,
        # Reward (neg logp).
        "L_reward_logp_B_T": L_reward_logp_B_T,
        "L_reward_logp": L_reward_logp,
        # Continues.
        "L_continue_B_T": L_continue_B_T,
        "L_continue": L_continue,

        # Dynamics loss.
        "L_dyn_B_T": L_dyn_B_T,
        "L_dyn": L_dyn,

        # Reproduction loss.
        "L_rep_B_T": L_rep_B_T,
        "L_rep": L_rep,

        # Total loss.
        "L_world_model_total_B_T": L_world_model_total_B_T,
        "L_world_model_total": L_world_model_total,
    }


@tf.function
def train_actor_and_critic_one_step(
    *,
    forward_train_outs,
    horizon_H,
    gamma,
    lambda_,
    actor_grad_clip,
    critic_grad_clip,
    dreamer_model,
    actor_optimizer,
    critic_optimizer,
    entropy_scale,
    return_normalization_decay,
    train_actor=True,
):
    # Compute losses.
    with tf.GradientTape(persistent=True) as tape:
        # Dream trajectories starting in all internal states (h+z) that were
        # computed during world model training.
        dream_data = dreamer_model.dream_trajectory(
            h=forward_train_outs["h_states_BxT"],
            z=forward_train_outs["z_states_BxT"],
            timesteps=horizon_H,
        )
        critic_loss_results = critic_loss(dream_data, gamma=gamma, lambda_=lambda_)
        if train_actor:
            actor_loss_results = actor_loss(
                dream_data=dream_data,
                value_targets=critic_loss_results["value_targets_B_H"],
                actor=dreamer_model.actor,
                entropy_scale=entropy_scale,
                return_normalization_decay=return_normalization_decay
            )

    results = critic_loss_results.copy()
    if train_actor:
        results.update(actor_loss_results)
        L_actor = results["L_actor"]

    results["dream_data"] = dream_data
    L_critic = results["L_critic"]

    # Get the gradients from the tape.
    if train_actor:
        actor_gradients = tape.gradient(
            L_actor,
            dreamer_model.actor.trainable_variables,
        )
    critic_gradients = tape.gradient(
        L_critic,
        dreamer_model.critic.trainable_variables,
    )

    # Clip all gradients.
    if train_actor:
        clipped_actor_gradients = []
        for grad in actor_gradients:
            clipped_actor_gradients.append(
                tf.clip_by_value(grad, -actor_grad_clip, actor_grad_clip)
            )
    clipped_critic_gradients = []
    for grad in critic_gradients:
        clipped_critic_gradients.append(
            tf.clip_by_value(grad, -critic_grad_clip, critic_grad_clip)
        )
    # Apply gradients to our models.
    if train_actor:
        actor_optimizer.apply_gradients(
            zip(clipped_actor_gradients, dreamer_model.actor.trainable_variables)
        )
    critic_optimizer.apply_gradients(
        zip(clipped_critic_gradients, dreamer_model.critic.trainable_variables)
    )

    # Update EMA weights of the critic.
    dreamer_model.critic.update_ema()

    if train_actor:
        results["actor_gradients"] = actor_gradients
    results["critic_gradients"] = critic_gradients

    return results
