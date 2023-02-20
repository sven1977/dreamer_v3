"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils.symlog import symlog
from utils.two_hot import two_hot


@tf.function
def world_model_prediction_losses(
        observations,
        rewards,
        terminateds,
        truncateds,
        B,
        T,
        forward_train_outs,
):
    obs_distr = forward_train_outs["obs_distribution"]
    # Learn to produce symlog'd observation predictions.
    # Fold time dim and flatten all other (image?) dims.
    observations = tf.reshape(observations, shape=[-1, int(np.prod(observations.shape.as_list()[2:]))])
    decoder_loss = - obs_distr.log_prob(symlog(observations))
    decoder_loss /= observations.shape.as_list()[1]
    #TODO: try MSE (instead of -log(p))
    #decoder_loss = tf.losses.mse(symlog(observations), obs_distr.loc)
    # Unfold time rank back in.
    decoder_loss = tf.reshape(decoder_loss, (B, T))

    # The FiniteDiscrete reward bucket distribution computed by our reward predictor.
    # [B x num_buckets].
    reward_distr = forward_train_outs["reward_distribution"]
    # Learn to produce symlog'd reward predictions.
    rewards = symlog(rewards)
    # Fold time dim.
    rewards = tf.reshape(rewards, shape=[-1])

    # A) Two-hot encode.
    two_hot_rewards = two_hot(rewards)
    # two_hot_rewards=[B*T, num_buckets]
    predicted_reward_log_probs = tf.math.log(reward_distr.probs)
    # predicted_reward_log_probs=[B*T, num_buckets]
    reward_loss_two_hot = - tf.reduce_sum(tf.multiply(two_hot_rewards, predicted_reward_log_probs), axis=-1)
    # Unfold time rank back in.
    reward_loss_two_hot = tf.reshape(reward_loss_two_hot, (B, T))

    # B) Simple neg log(p) on distribution, NOT using two-hot.
    reward_loss_logp = - reward_distr.log_prob(rewards)
    # Unfold time rank back in.
    reward_loss_logp = tf.reshape(reward_loss_logp, (B, T))

    # Continue predictor loss.
    continues = tf.logical_not(tf.logical_or(terminateds, truncateds))
    # Probabilities that episode continues, computed by our continue predictor.
    # [B]
    continue_distr = forward_train_outs["continue_distribution"]
    # -log(p) loss
    # Fold time dim.
    continues = tf.reshape(continues, shape=[-1])
    continue_loss = - continue_distr.log_prob(continues)
    # Unfold time rank back in.
    continue_loss = tf.reshape(continue_loss, (B, T))

    return {
        "decoder_loss": decoder_loss,
        "reward_loss_two_hot": reward_loss_two_hot,
        "reward_loss_logp": reward_loss_logp,
        "continue_loss": continue_loss,
        "total_loss": decoder_loss + reward_loss_two_hot + continue_loss,
    }


@tf.function
def world_model_dynamics_and_representation_loss(forward_train_outs, B, T):
    # Actual distribution over stochastic internal states (z) produced by the encoder.
    z_distr_encoder = forward_train_outs["z_distribution_encoder"]
    # Actual distribution over stochastic internal states (z) produced by the
    # dynamics network.
    z_distr_dynamics = forward_train_outs["z_distribution_dynamics"]

    # Stop gradient for encoder's z-outputs:
    sg_z_distr_encoder = tfp.distributions.Categorical(
        probs=tf.stop_gradient(z_distr_encoder.probs)
    )
    # Stop gradient for dynamics model's z-outputs:
    sg_z_distr_dynamics = tfp.distributions.Categorical(
        probs=tf.stop_gradient(z_distr_dynamics.probs)
    )

    # Implement free bits. According to [1]:
    # "To avoid a degenerate solution where the dynamics are trivial to predict but
    # contain not enough information about the inputs, we employ free bits by clipping
    # the dynamics and representation losses below the value of 1 nat ≈ 1.44 bits. This
    # disables them while they are already minimized well to focus the world model
    # on its prediction loss"
    L_dyn = tf.math.maximum(
        1.0,
        # Sum KL over all `num_categoricals` as these are independent.
        # This is the same thing that a tfp.distributions.Independent() distribution
        # with an underlying set of different Categoricals would do.
        tf.reduce_sum(
            tfp.distributions.kl_divergence(sg_z_distr_encoder, z_distr_dynamics),
            axis=-1,
        ),
    )
    # Unfold time rank back in.
    L_dyn = tf.reshape(L_dyn, (B, T))

    L_rep = tf.math.maximum(
        1.0,
        # Sum KL over all `num_categoricals` as these are independent.
        # This is the same thing that a tfp.distributions.Independent() distribution
        # with an underlying set of different Categoricals would do.
        tf.reduce_sum(
            tfp.distributions.kl_divergence(z_distr_encoder, sg_z_distr_dynamics),
            axis=-1,
        ),
    )
    # Unfold time rank back in.
    L_rep = tf.reshape(L_rep, (B, T))

    return L_dyn, L_rep
