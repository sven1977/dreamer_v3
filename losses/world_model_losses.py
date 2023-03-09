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
        rewards,
        continues,
        B,
        T,
        forward_train_outs,
):
    # Learn to produce symlog'd observation predictions.
    # If symlog is disabled (e.g. for uint8 image inputs), `obs_symlog_BxT` is the same
    # as `obs_BxT`.
    obs_BxT = forward_train_outs["sampled_obs_symlog_BxT"]
    obs_distr = forward_train_outs["obs_distribution_BxT"]
    # Fold time dim and flatten all other (image?) dims.
    obs_BxT = tf.reshape(
        obs_BxT, shape=[-1, int(np.prod(obs_BxT.shape.as_list()[1:]))]
    )

    # Neg logp loss.
    #decoder_loss = - obs_distr.log_prob(observations)
    #decoder_loss /= observations.shape.as_list()[1]
    # Squared diff loss w/ sum(!) over all (already folded) obs dims.
    decoder_loss_BxT = tf.reduce_sum(tf.math.square(obs_distr.loc - obs_BxT), axis=-1)

    # Unfold time rank back in.
    decoder_loss_B_T = tf.reshape(decoder_loss_BxT, (B, T))

    # The FiniteDiscrete reward bucket distribution computed by our reward predictor.
    # [B x num_buckets].
    reward_logits_BxT = forward_train_outs["reward_logits_BxT"]
    # Learn to produce symlog'd reward predictions.
    rewards = symlog(rewards)
    # Fold time dim.
    rewards = tf.reshape(rewards, shape=[-1])

    # A) Two-hot encode.
    two_hot_rewards = two_hot(rewards)
    # two_hot_rewards=[B*T, num_buckets]
    #predicted_reward_log_probs = tf.math.log(reward_distr.probs)
    #predicted_reward_probs = reward_distr.probs
    # predicted_reward_log_probs=[B*T, num_buckets]
    #reward_loss_two_hot = - tf.reduce_sum(tf.multiply(two_hot_rewards, predicted_reward_log_probs), axis=-1)
    #reward_loss_two_hot_BxT = - tf.math.log(tf.reduce_sum(tf.multiply(two_hot_rewards, predicted_reward_probs), axis=-1))
    reward_log_pred_BxT = reward_logits_BxT - tf.math.reduce_logsumexp(reward_logits_BxT, axis=-1, keepdims=True)
    # Multiply with two-hot targets and neg.
    reward_loss_two_hot_BxT = - tf.reduce_sum(reward_log_pred_BxT * two_hot_rewards, axis=-1)
    # Unfold time rank back in.
    reward_loss_two_hot_B_T = tf.reshape(reward_loss_two_hot_BxT, (B, T))

    # B) Simple neg log(p) on distribution, NOT using two-hot.
    #reward_loss_logp_BxT = - reward_distr.log_prob(rewards)
    ## Unfold time rank back in.
    #reward_loss_logp_B_T = tf.reshape(reward_loss_logp_BxT, (B, T))

    # Probabilities that episode continues, computed by our continue predictor.
    # [B]
    continue_distr = forward_train_outs["continue_distribution_BxT"]
    # -log(p) loss
    # Fold time dim.
    continues = tf.reshape(continues, shape=[-1])
    continue_loss_BxT = - continue_distr.log_prob(continues)
    # Unfold time rank back in.
    continue_loss_B_T = tf.reshape(continue_loss_BxT, (B, T))

    return {
        "decoder_loss_B_T": decoder_loss_B_T,
        "reward_loss_two_hot_B_T": reward_loss_two_hot_B_T,
        #"reward_loss_logp_B_T": reward_loss_logp_B_T,
        "continue_loss_B_T": continue_loss_B_T,
        "prediction_loss_B_T": decoder_loss_B_T + reward_loss_two_hot_B_T + continue_loss_B_T,
    }


@tf.function
def world_model_dynamics_and_representation_loss(forward_train_outs, B, T):
    # Actual distribution over stochastic internal states (z) produced by the encoder.
    z_probs_encoder_BxT = forward_train_outs["z_probs_encoder_BxT"]
    z_distr_encoder_BxT = tfp.distributions.Independent(
        tfp.distributions.OneHotCategorical(probs=z_probs_encoder_BxT),
        reinterpreted_batch_ndims=1,
    )

    # Actual distribution over stochastic internal states (z) produced by the
    # dynamics network.
    z_probs_dynamics_BxT = forward_train_outs["z_probs_dynamics_BxT"]
    z_distr_dynamics_BxT = tfp.distributions.Independent(
        tfp.distributions.OneHotCategorical(probs=z_probs_dynamics_BxT),
        reinterpreted_batch_ndims=1,
    )

    # Stop gradient for encoder's z-outputs:
    sg_z_distr_encoder_BxT = tfp.distributions.Independent(
        tfp.distributions.OneHotCategorical(
            probs=tf.stop_gradient(z_probs_encoder_BxT)
        ),
        reinterpreted_batch_ndims=1,
    )
    # Stop gradient for dynamics model's z-outputs:
    sg_z_distr_dynamics_BxT = tfp.distributions.Independent(
        tfp.distributions.OneHotCategorical(
            probs=tf.stop_gradient(z_probs_dynamics_BxT)
        ),
        reinterpreted_batch_ndims=1,
    )

    # Implement free bits. According to [1]:
    # "To avoid a degenerate solution where the dynamics are trivial to predict but
    # contain not enough information about the inputs, we employ free bits by clipping
    # the dynamics and representation losses below the value of 1 nat ≈ 1.44 bits. This
    # disables them while they are already minimized well to focus the world model
    # on its prediction loss"
    L_dyn_BxT = tf.math.maximum(
        1.0,
        ## Sum KL over all `num_categoricals` as these are independent.
        ## This is the same thing that a tfp.distributions.Independent() distribution
        ## with an underlying set of different Categoricals would do.
        #tf.reduce_sum(
        tfp.distributions.kl_divergence(sg_z_distr_encoder_BxT, z_distr_dynamics_BxT),
        #    axis=-1,
        #),
    )
    # Unfold time rank back in.
    L_dyn_B_T = tf.reshape(L_dyn_BxT, (B, T))

    L_rep_BxT = tf.math.maximum(
        1.0,
        ## Sum KL over all `num_categoricals` as these are independent.
        ## This is the same thing that a tfp.distributions.Independent() distribution
        ## with an underlying set of different Categoricals would do.
        #tf.reduce_sum(
        tfp.distributions.kl_divergence(z_distr_encoder_BxT, sg_z_distr_dynamics_BxT),
        #    axis=-1,
        #),
    )
    # Unfold time rank back in.
    L_rep_B_T = tf.reshape(L_rep_BxT, (B, T))

    return L_dyn_B_T, L_rep_B_T
