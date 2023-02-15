import tensorflow as tf

from utils.symlog import inverse_symlog


def summarize_forward_train_outs_vs_samples(
    forward_train_outs,
    sample,
    batch_size_B,
    batch_length_T,
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
        computed_float_obs_B_T_dims=tf.reshape(inverse_symlog(
            forward_train_outs["obs_distribution"].loc
        ), shape=(batch_size_B, batch_length_T) + sample["obs"].shape[2:]),
        sampled_obs_B_T_dims=sample["obs"],
        B=batch_size_B,
        T=batch_length_T,
        descr="predicted(posterior)",
    )

    predicted_rewards = tf.reshape(
        inverse_symlog(forward_train_outs["reward_distribution"].mean()),
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
        forward_train_outs["continue_distribution"].mode(),
        shape=(batch_size_B, batch_length_T),
    )
    _summarize_continues(
        computed_continues_B_T=predicted_continues,
        sampled_terminateds_B_T=sample["terminateds"],
        sampled_truncateds_B_T=sample["truncateds"],
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
):
    # Obs MSE.
    # Compute actual observations using h and z and the decoder net.
    # Note that the last h-state (T+1) is NOT used here as it's already part of
    # a new trajectory.
    _, dreamed_obs_distr = dreamer_model.world_model.decoder(
        h=tf.reshape(
            dream_data["h_states_t1_to_T+1"][:, :-1],
            shape=(batch_size_B * dreamed_T, -1),
        ),
        z=tf.reshape(
            dream_data["z_states_prior_t1_to_T"],
            shape=(
                (batch_size_B * dreamed_T)
                + dream_data["z_states_prior_t1_to_T"].shape[2:]
            ),
        ),
    )
    dreamed_obs = tf.reshape(
        # Use mean() of the Gaussian, no sample!
        inverse_symlog(dreamed_obs_distr.loc),
        shape=(batch_size_B, dreamed_T) + sample["obs"].shape[2:],
    )
    # Observation MSE and - if applicable - images comparisons.
    mse_sampled_vs_dreamed_obs = _summarize_obs(
        computed_float_obs_B_T_dims=dreamed_obs,
        sampled_obs_B_T_dims=sample["obs"][:, burn_in_T:burn_in_T + dreamed_T],
        B=batch_size_B,
        T=dreamed_T,
        descr="dreamed(prior)",
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
        sampled_terminateds_B_T=(
            sample["terminateds"][:, burn_in_T:burn_in_T + dreamed_T]
        ),
        sampled_truncateds_B_T=(
            sample["truncateds"][:, burn_in_T:burn_in_T + dreamed_T]
        ),
        B=batch_size_B,
        T=dreamed_T,
        descr="dreamed(prior)",
    )
    return mse_sampled_vs_dreamed_obs


def summarize_world_model_losses(world_model_train_results):
    res = world_model_train_results

    tf.summary.histogram("L_pred_BxT", world_model_train_results["L_pred_BxT"])
    tf.summary.scalar("L_pred", world_model_train_results["L_pred"])

    tf.summary.histogram("L_decoder_BxT", world_model_train_results["L_decoder_BxT"])
    tf.summary.scalar("L_decoder", world_model_train_results["L_decoder"])

    # Two-hot reward loss.
    tf.summary.histogram(
        "L_reward_two_hot_BxT", world_model_train_results["L_reward_two_hot_BxT"]
    )
    tf.summary.scalar(
        "L_reward_two_hot", world_model_train_results["L_reward_two_hot"]
    )
    # TEST: Out of interest, compare with simplge -log(p) loss for individual
    # rewards using the FiniteDiscrete distribution. This should be very close
    # to the two-hot reward loss.
    tf.summary.histogram(
        "L_reward_logp_BxT", world_model_train_results["L_reward_logp_BxT"]
    )
    tf.summary.scalar(
        "L_reward_logp", world_model_train_results["L_reward_logp"]
    )

    tf.summary.histogram("L_continue_BxT", world_model_train_results["L_continue_BxT"])
    tf.summary.scalar("L_continue", world_model_train_results["L_continue"])

    tf.summary.histogram("L_dyn_BxT", world_model_train_results["L_dyn_BxT"])
    tf.summary.scalar("L_dyn", world_model_train_results["L_dyn"])

    tf.summary.histogram("L_rep_BxT", world_model_train_results["L_rep_BxT"])
    tf.summary.scalar("L_rep", world_model_train_results["L_rep"])

    # Total loss.
    tf.summary.histogram("L_world_model_total_BxT", world_model_train_results["L_world_model_total_BxT"])
    tf.summary.scalar("L_world_model_total", world_model_train_results["L_world_model_total"])


def _summarize_obs(*, computed_float_obs_B_T_dims, sampled_obs_B_T_dims, B, T, descr):
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
        descr: A string used to describe the computed data tobe used in the TB
            summaries.
    """
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
    sampled_terminateds_B_T,
    sampled_truncateds_B_T,
    B,
    T,
    descr,
):
    # Continue MSE.
    mse_sampled_vs_computed_continues = tf.losses.mse(
        tf.expand_dims(computed_continues_B_T, axis=-1),
        tf.expand_dims(tf.cast(
            tf.logical_not(
                tf.logical_or(
                    sampled_terminateds_B_T,
                    sampled_truncateds_B_T,
                )
            ),
            dtype=tf.float32,
        ), axis=-1),
    )
    mse_sampled_vs_computed_continues = tf.reduce_mean(
        tf.reduce_sum(mse_sampled_vs_computed_continues, axis=1)
    )
    tf.summary.scalar(
        f"MEAN(SUM(mse,T={T}),B={B})_sampled_vs_{descr}_continues",
        mse_sampled_vs_computed_continues,
    )

