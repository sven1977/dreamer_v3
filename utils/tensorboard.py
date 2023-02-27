import tensorflow as tf

from utils.symlog import inverse_symlog


def reconstruct_obs_from_h_and_z(
    h_t1_to_Tp1,
    z_t1_to_T,
    dreamer_model,
    obs_dims_shape,
):
    """Returns """
    shape = tf.shape(z_t1_to_T)
    B = shape[0]
    T = shape[1]
    # Compute actual observations using h and z and the decoder net.
    # Note that the last h-state (T+1) is NOT used here as it's already part of
    # a new trajectory.
    _, reconstructed_obs_distr_BxT = dreamer_model.world_model.decoder(
        # Fold time rank.
        h=tf.reshape(h_t1_to_Tp1[:, :-1], shape=(B * T, -1)),
        z=tf.reshape(z_t1_to_T, shape=(B * T,) + z_t1_to_T.shape[2:]),
    )
    # Use mean() of the Gaussian, no sample!
    loc = reconstructed_obs_distr_BxT.loc
    # Unfold time rank again.
    reconstructed_obs_B_T = tf.reshape(loc, shape=(B, T) + obs_dims_shape)
    # Return inverse symlog'd (real env obs space) reconstructed observations.
    return reconstructed_obs_B_T


def summarize_forward_train_outs_vs_samples(
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
        computed_float_obs_B_T_dims=tf.reshape(
            forward_train_outs["obs_distribution_BxT"].loc,
            shape=(batch_size_B, batch_length_T) + sample["obs"].shape[2:],
        ),
        sampled_obs_B_T_dims=sample["obs"],
        B=batch_size_B,
        T=batch_length_T,
        descr="predicted(posterior)",
        symlog_obs=symlog_obs,
    )
    predicted_rewards = tf.reshape(
        inverse_symlog(forward_train_outs["reward_distribution_BxT"].mean()),
        shape=(batch_size_B, batch_length_T),
    )
    _summarize_rewards(
        computed_rewards_B_T=predicted_rewards,
        sampled_rewards_B_T=sample["rewards"],
        B=batch_size_B,
        T=batch_length_T,
        descr="predicted(posterior)",
    )
    tf.summary.histogram("sampled_rewards", sample["rewards"])
    tf.summary.histogram("predicted(posterior)_rewards", predicted_rewards)

    predicted_continues = tf.reshape(
        forward_train_outs["continue_distribution_BxT"].mode(),
        shape=(batch_size_B, batch_length_T),
    )
    _summarize_continues(
        computed_continues_B_T=predicted_continues,
        sampled_continues_B_T=sample["continues"],
        B=batch_size_B,
        T=batch_length_T,
        descr="predicted(posterior)",
    )


def summarize_dreamed_trajectory_vs_samples(
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
        computed_float_obs_B_T_dims=dreamed_obs,
        sampled_obs_B_T_dims=sample["obs"][:, burn_in_T:burn_in_T + dreamed_T],
        B=batch_size_B,
        T=dreamed_T,
        descr="dreamed(prior)",
        symlog_obs=symlog_obs,
    )

    # Reward MSE.
    _summarize_rewards(
        computed_rewards_B_T=dream_data["rewards_dreamed_t1_to_T"],
        sampled_rewards_B_T=sample["rewards"][:, burn_in_T:burn_in_T + dreamed_T],
        B=batch_size_B,
        T=dreamed_T,
        descr="dreamed(prior)",
    )

    # Continues MSE.
    _summarize_continues(
        computed_continues_B_T=dream_data["continues_dreamed_t1_to_T"],
        sampled_continues_B_T=sample["continues"][:, burn_in_T:burn_in_T + dreamed_T],
        B=batch_size_B,
        T=dreamed_T,
        descr="dreamed(prior)",
    )
    return mse_sampled_vs_dreamed_obs


def summarize_world_model_losses(world_model_train_results):
    tf.summary.histogram("L_pred_B_T", world_model_train_results["L_pred_B_T"])
    tf.summary.scalar("L_pred", world_model_train_results["L_pred"])

    tf.summary.histogram("L_decoder_B_T", world_model_train_results["L_decoder_B_T"])
    tf.summary.scalar("L_decoder", world_model_train_results["L_decoder"])

    # Two-hot reward loss.
    tf.summary.histogram(
        "L_reward_two_hot_B_T", world_model_train_results["L_reward_two_hot_B_T"]
    )
    tf.summary.scalar(
        "L_reward_two_hot", world_model_train_results["L_reward_two_hot"]
    )
    # TEST: Out of interest, compare with simplge -log(p) loss for individual
    # rewards using the FiniteDiscrete distribution. This should be very close
    # to the two-hot reward loss.
    tf.summary.histogram(
        "L_reward_logp_B_T", world_model_train_results["L_reward_logp_B_T"]
    )
    tf.summary.scalar(
        "L_reward_logp", world_model_train_results["L_reward_logp"]
    )

    tf.summary.histogram("L_continue_B_T", world_model_train_results["L_continue_B_T"])
    tf.summary.scalar("L_continue", world_model_train_results["L_continue"])

    tf.summary.histogram("L_dyn_B_T", world_model_train_results["L_dyn_B_T"])
    tf.summary.scalar("L_dyn", world_model_train_results["L_dyn"])

    tf.summary.histogram("L_rep_B_T", world_model_train_results["L_rep_B_T"])
    tf.summary.scalar("L_rep", world_model_train_results["L_rep"])

    # Total loss.
    tf.summary.histogram("L_world_model_total_B_T", world_model_train_results["L_world_model_total_B_T"])
    tf.summary.scalar("L_world_model_total", world_model_train_results["L_world_model_total"])


def _summarize_obs(*, computed_float_obs_B_T_dims, sampled_obs_B_T_dims, B, T, descr, symlog_obs):
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
    if symlog_obs:
        computed_float_obs_B_T_dims = inverse_symlog(computed_float_obs_B_T_dims)

    # MSE is the mean over all feature dimensions.
    # Images: Flatten image dimensions (w, h, C); Vectors: Mean over all items, etc..
    # Then sum over time-axis and mean over batch-axis.
    mse_sampled_vs_computed_obs = tf.losses.mse(
        tf.reshape(computed_float_obs_B_T_dims, (B, T, -1)),
        tf.reshape(tf.cast(sampled_obs_B_T_dims, tf.float32), shape=(B, T, -1)),
    )
    mse_sampled_vs_computed_obs = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_obs, axis=1)
    )
    tf.summary.scalar(
        f"MEAN(SUM(mse,T={T}),B={B})_sampled_vs_{descr}_obs",
        mse_sampled_vs_computed_obs,
    )

    # Images: Create image summary, comparing computed images with actual sampled ones.
    # Note: We only use images here from the first (0-index) batch item.
    if len(sampled_obs_B_T_dims.shape) in [2+2, 2+3]:
        # Restore image pixels from normalized (non-symlog'd) data.
        if not symlog_obs:
            computed_float_obs_B_T_dims = (computed_float_obs_B_T_dims + 1.0) * 128

        computed_images = tf.cast(
            tf.clip_by_value(computed_float_obs_B_T_dims, 0.0, 255.0), tf.uint8
        )
        # Concat sampled and computed images along the height axis (2) such that
        # real images show on top of respective predicted ones.
        # (B, w, h, C)
        sampled_vs_computed_images = tf.concat(
            [computed_images[0], sampled_obs_B_T_dims[0]], axis=1,
        )
        tf.summary.image(
            f"sampled_vs_{descr}_images[0th batch item]",
            (
                tf.expand_dims(sampled_vs_computed_images, -1)
                if len(sampled_obs_B_T_dims.shape) == 2+2
                else sampled_vs_computed_images
            ),
            max_outputs=20,
        )

    return mse_sampled_vs_computed_obs


def _summarize_rewards(*, computed_rewards_B_T, sampled_rewards_B_T, B, T, descr):
    mse_sampled_vs_computed_rewards = tf.losses.mse(
        tf.expand_dims(computed_rewards_B_T, axis=-1),
        tf.expand_dims(sampled_rewards_B_T, axis=-1),
    )
    mse_sampled_vs_computed_rewards = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_rewards, axis=1)
    )
    tf.summary.scalar(
        f"MEAN(SUM(mse,T={T}),B={B})_sampled_vs_{descr}_rewards",
        mse_sampled_vs_computed_rewards,
    )


def _summarize_continues(
    *,
    computed_continues_B_T,
    sampled_continues_B_T,
    B,
    T,
    descr,
):
    # Continue MSE.
    mse_sampled_vs_computed_continues = tf.losses.mse(
        tf.expand_dims(computed_continues_B_T, axis=-1),
        tf.expand_dims(tf.cast(sampled_continues_B_T, dtype=tf.float32), axis=-1),
    )
    mse_sampled_vs_computed_continues = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_continues, axis=1)
    )
    tf.summary.scalar(
        f"MEAN(SUM(mse,T={T}),B={B})_sampled_vs_{descr}_continues",
        mse_sampled_vs_computed_continues,
    )

