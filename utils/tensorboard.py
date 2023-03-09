import tensorflow as tf
import tree  # pip install dm_tree

from utils.symlog import inverse_symlog


def _summarize(tbx_writer, step, keys_to_log, results):
    results = tree.map_structure(
        lambda s: s.numpy() if tf.is_tensor(s) else s,
        results,
    )

    for k in keys_to_log:
        if results[k].shape == ():
            tbx_writer.add_scalar(k, results[k], global_step=step)
        if results[k].shape == ():
            tbx_writer.add_histogram(k, results[k], global_step=step)


def reconstruct_obs_from_h_and_z(
    h_t0_to_H,
    z_t0_to_H,
    dreamer_model,
    obs_dims_shape,
):
    """Returns """
    shape = tf.shape(h_t0_to_H)
    T = shape[0]  # inputs are time-major
    B = shape[1]
    # Compute actual observations using h and z and the decoder net.
    # Note that the last h-state (T+1) is NOT used here as it's already part of
    # a new trajectory.
    _, reconstructed_obs_distr_TxB = dreamer_model.world_model.decoder(
        # Fold time rank.
        h=tf.reshape(h_t0_to_H, shape=(T * B, -1)),
        z=tf.reshape(z_t0_to_H, shape=(T * B,) + z_t0_to_H.shape[2:]),
    )
    # Use mean() of the Gaussian, no sample!
    loc = reconstructed_obs_distr_TxB.loc
    # Unfold time rank again.
    reconstructed_obs_T_B = tf.reshape(loc, shape=(T, B) + obs_dims_shape)
    # Return inverse symlog'd (real env obs space) reconstructed observations.
    return reconstructed_obs_T_B


def summarize_forward_train_outs_vs_samples(
    *,
    tbx_writer,
    step,
    forward_train_outs,
    sample,
    batch_size_B,
    batch_length_T,
    symlog_obs: bool = True,
):
    """Summarizes sampled data (from the replay buffer) vs world-model predictions.

    World model predictions are based on the posterior states (z computed from actual
    observation encoder input + the current h-states).

    Observations: Computes MSE (sampled vs predicted/recreated) over all features.
    For image observations, also creates direct image comparisons (sampled images
    vs predicted (posterior) ones).
    Rewards: Compute MSE (sampled vs predicted).
    Continues: Compute MSE (sampled vs predicted).

    Args:
        forward_train_outs: The results dict returned by the world model's
            `forward_train` method.
        sample: The sampled data (dict) from the replay buffer. Already tf-tensor
            converted.
        batch_size_B: The batch size (B). This is the number of trajectories sampled
            from the buffer.
        batch_length_T: The batch length (T). This is the length of an individual
            trajectory sampled from the buffer.
    """
    _summarize_obs(
        tbx_writer=tbx_writer,
        step=step,
        computed_float_obs_B_T_dims=tf.reshape(
            forward_train_outs["obs_distribution_BxT"].loc,
            shape=(batch_size_B, batch_length_T) + sample["obs"].shape[2:],
        ),
        sampled_obs_B_T_dims=sample["obs"],
        descr_prefix="WORLD_MODEL",
        descr_obs=f"predicted_posterior_T{batch_length_T}",
        symlog_obs=symlog_obs,
    )
    predicted_rewards = tf.reshape(
        inverse_symlog(forward_train_outs["rewards_BxT"]),
        shape=(batch_size_B, batch_length_T),
    )
    _summarize_rewards(
        tbx_writer=tbx_writer,
        step=step,
        computed_rewards_B_T=predicted_rewards,
        sampled_rewards_B_T=sample["rewards"],
        descr_prefix="WORLD_MODEL",
        descr_reward="predicted_posterior",
    )
    tbx_writer.add_histogram(
        "sampled_rewards", sample["rewards"].numpy(), global_step=step
    )
    tbx_writer.add_histogram(
        "predicted_posterior_rewards", predicted_rewards.numpy(), global_step=step
    )

    predicted_continues = tf.reshape(
        forward_train_outs["continues_BxT"],
        shape=(batch_size_B, batch_length_T),
    )
    _summarize_continues(
        tbx_writer=tbx_writer,
        step=step,
        computed_continues_B_T=predicted_continues,
        sampled_continues_B_T=(1.0 - sample["is_terminated"]),
        descr_prefix="WORLD_MODEL",
        descr_cont="predicted_posterior",
    )


def summarize_actor_train_results(*, tbx_writer, step, actor_critic_train_results):
    keys_to_log = [
        "ACTOR_L_total",
        "ACTOR_action_entropy",
        "ACTOR_L_reinforce_term",
        "ACTOR_L_entropy_term",
        "ACTOR_scaled_value_targets_H_B",
        "ACTOR_L_logp_H_B",
        "ACTOR_value_targets_pct95_ema",
        "ACTOR_value_targets_pct5_ema",
        "ACTOR_gradients_maxabs",
        "ACTOR_gradients_clipped_by_glob_norm_maxabs",
    ]

    _summarize(tbx_writer, step, keys_to_log, actor_critic_train_results)


def summarize_critic_train_results(*, tbx_writer, step, actor_critic_train_results):
    keys_to_log = [
        # TODO: Move this to generic function as value targets are also important for actor
        #  loss.
        "VALUE_TARGETS_H_B",
        "VALUE_TARGETS_symlog_H_B",

        "CRITIC_L_total",
        "CRITIC_L_neg_logp_of_value_targets_H_B",
        "CRITIC_L_slow_critic_regularization_H_B",
        "CRITIC_gradients_maxabs",
        "CRITIC_gradients_clipped_by_glob_norm_maxabs",
    ]

    _summarize(tbx_writer, step, keys_to_log, actor_critic_train_results)


def summarize_dreamed_eval_trajectory_vs_samples(
    *,
    tbx_writer,
    step,
    dream_data,
    sample,
    batch_size_B,
    burn_in_T,
    dreamed_T,
    dreamer_model,
    symlog_obs: bool = True,
):
    # Obs MSE.
    dreamed_obs = reconstruct_obs_from_h_and_z(
        h_t1_to_Tp1=dream_data["h_states_t1_to_Tp1"],
        z_t1_to_T=dream_data["z_states_prior_t1_to_T"],
        dreamer_model=dreamer_model,
        obs_dims_shape=sample["obs"].shape[2:],
    )
    # Observation MSE and - if applicable - images comparisons.
    mse_sampled_vs_dreamed_obs = _summarize_obs(
        tbx_writer=tbx_writer,
        step=step,
        computed_float_obs_B_T_dims=dreamed_obs,
        sampled_obs_B_T_dims=sample["obs"][:, burn_in_T:burn_in_T + dreamed_T],
        descr_prefix="EVAL",
        descr_obs=f"dreamed_prior_H{dreamed_T}",
        symlog_obs=symlog_obs,
    )

    # Reward MSE.
    _summarize_rewards(
        tbx_writer=tbx_writer,
        step=step,
        computed_rewards_B_T=dream_data["rewards_dreamed_t1_to_T"],
        sampled_rewards_B_T=sample["rewards"][:, burn_in_T:burn_in_T + dreamed_T],
        descr_prefix="EVAL",
        descr_reward=f"dreamed_prior_H{dreamed_T}",
    )

    # Continues MSE.
    _summarize_continues(
        tbx_writer=tbx_writer,
        step=step,
        computed_continues_B_T=dream_data["continues_dreamed_t1_to_T"],
        sampled_continues_B_T=(1.0 - sample["is_terminated"])[:, burn_in_T:burn_in_T + dreamed_T],
        descr_prefix="EVAL",
        descr_cont=f"dreamed_prior_H{dreamed_T}",
    )
    return mse_sampled_vs_dreamed_obs


def summarize_world_model_train_results(*, tbx_writer, step, world_model_train_results):
    # TODO: Move to returned train_one_step results.
    tbx_writer.add_scalar("WORLD_MODEL_initial_h_sum_abs", tf.reduce_sum(
        tf.math.abs(world_model_train_results["WORLD_MODEL_learned_initial_h"].numpy())
    ).numpy(), global_step=step)

    keys_to_log = [
        # Learned initial state.
        "WORLD_MODEL_learned_initial_h",

        # Losses.
        "WORLD_MODEL_L_prediction_B_T",
        "WORLD_MODEL_L_prediction",
        "WORLD_MODEL_L_decoder_B_T",
        "WORLD_MODEL_L_decoder",
        "WORLD_MODEL_L_reward_B_T",
        "WORLD_MODEL_L_reward",
        "WORLD_MODEL_L_continue_B_T",
        "WORLD_MODEL_L_continue",

        "WORLD_MODEL_L_dynamics_B_T",
        "WORLD_MODEL_L_dynamics",

        "WORLD_MODEL_L_representation_B_T",
        "WORLD_MODEL_L_representation",

        "WORLD_MODEL_L_total_B_T",
        "WORLD_MODEL_L_total",

        # Gradients.
        "WORLD_MODEL_gradients_maxabs",
        "WORLD_MODEL_gradients_clipped_by_glob_norm_maxabs",
    ]

    _summarize(tbx_writer, step, keys_to_log, world_model_train_results)


def _summarize_obs(
    *,
    tbx_writer,
    step,
    computed_float_obs_B_T_dims,
    sampled_obs_B_T_dims,
    descr_prefix=None,
    descr_obs,
    symlog_obs,
):
    """Summarizes computed- vs sampled observations: MSE and (if applicable) images.

    Args:
        computed_float_obs_B_T_dims: Computed float observations
            (not clipped, not cast'd). Shape=(B, T, [dims ...]).
        sampled_obs_B_T_dims: Sampled observations (as-is from the environment, meaning
            this could be uint8, 0-255 clipped images). Shape=(B, T, [dims ...]).
        B: The batch size B (see shapes of `computed_float_obs_B_T_dims` and
            `sampled_obs_B_T_dims` above).
        T: The batch length T (see shapes of `computed_float_obs_B_T_dims` and
            `sampled_obs_B_T_dims` above).
        descr: A string used to describe the computed data to be used in the TB
            summaries.
    """
    descr_prefix = (descr_prefix + "_") if descr_prefix else ""

    if symlog_obs:
        computed_float_obs_B_T_dims = inverse_symlog(computed_float_obs_B_T_dims)

    # MSE is the mean over all feature dimensions.
    # Images: Flatten image dimensions (w, h, C); Vectors: Mean over all items, etc..
    # Then sum over time-axis and mean over batch-axis.
    mse_sampled_vs_computed_obs = tf.math.square(
        computed_float_obs_B_T_dims - tf.cast(sampled_obs_B_T_dims, tf.float32)
    )
    mse_sampled_vs_computed_obs = tf.reduce_mean(mse_sampled_vs_computed_obs)
    tbx_writer.add_scalar(
        f"{descr_prefix}sampled_vs_{descr_obs}_obs_mse",
        mse_sampled_vs_computed_obs.numpy(),
        global_step=step,
    )

    # Videos: Create summary, comparing computed images with actual sampled ones.
    if len(sampled_obs_B_T_dims.shape) in [2+2, 2+3]:
        # Restore image pixels from normalized (non-symlog'd) data.
        if not symlog_obs:
            computed_float_obs_B_T_dims = (computed_float_obs_B_T_dims + 1.0) * 128
            sampled_obs_B_T_dims = (sampled_obs_B_T_dims + 1.0) * 128
            sampled_obs_B_T_dims = tf.cast(
                tf.clip_by_value(sampled_obs_B_T_dims, 0.0, 255.0), tf.uint8
            )
        computed_images = tf.cast(
            tf.clip_by_value(computed_float_obs_B_T_dims, 0.0, 255.0), tf.uint8
        )
        # Concat sampled and computed images along the height axis (2) such that
        # real images show below respective predicted ones.
        # (B, T, h, w, C)
        sampled_vs_computed_images = tf.concat(
            [computed_images, sampled_obs_B_T_dims], axis=2,
        )
        # Add grayscale dim, if necessary.
        if len(sampled_obs_B_T_dims.shape) == 2 + 2:
            sampled_vs_computed_images = tf.expand_dims(sampled_vs_computed_images, -1)

        tbx_writer.add_video(
            f"{descr_prefix}sampled_vs_{descr_obs}_videos",
            sampled_vs_computed_images.numpy(),
            dataformats="NTHWC",
            global_step=step,
        )

    return mse_sampled_vs_computed_obs


def _summarize_rewards(
    *,
    tbx_writer,
    step,
    computed_rewards_B_T,
    sampled_rewards_B_T,
    descr_prefix=None,
    descr_reward,
):
    descr_prefix = (descr_prefix + "_") if descr_prefix else ""
    mse_sampled_vs_computed_rewards = tf.losses.mse(
        tf.expand_dims(computed_rewards_B_T, axis=-1),
        tf.expand_dims(sampled_rewards_B_T, axis=-1),
    )
    mse_sampled_vs_computed_rewards = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_rewards, axis=1)
    )
    tbx_writer.add_scalar(
        f"{descr_prefix}sampled_vs_{descr_reward}_rewards_mse",
        mse_sampled_vs_computed_rewards.numpy(),
        global_step=step,
    )


def _summarize_continues(
    *,
    tbx_writer,
    step,
    computed_continues_B_T,
    sampled_continues_B_T,
    descr_prefix=None,
    descr_cont,
):
    descr_prefix = (descr_prefix + "_") if descr_prefix else ""
    # Continue MSE.
    mse_sampled_vs_computed_continues = tf.losses.mse(
        tf.expand_dims(computed_continues_B_T, axis=-1),
        tf.expand_dims(tf.cast(sampled_continues_B_T, dtype=tf.float32), axis=-1),
    )
    mse_sampled_vs_computed_continues = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_continues, axis=1)
    )
    tbx_writer.add_scalar(
        f"{descr_prefix}sampled_vs_{descr_cont}_continues_mse",
        mse_sampled_vs_computed_continues.numpy(),
        global_step=step,
    )

