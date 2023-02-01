import tensorflow as tf
import tensorflow_probability as tfp

from utils.symlog import symlog


def world_model_prediction_loss(
        observations,
        rewards,
        terminateds,
        truncateds,
        forward_train_outs,
):
    obs_distr = forward_train_outs["obs_distribution"]
    # Learn to produce symlog'd observation predictions.
    decoder_loss = - obs_distr.log_prob(symlog(observations))

    # Probabilities of the individual reward value buckets computed by our reward
    # predictor.
    # [B x num_buckets].
    reward_distr = forward_train_outs["reward_distribution"]
    # Learn to produce symlog'd reward predictions.
    reward_loss = - reward_distr.log_prob(symlog(rewards))

    # Continue predictor loss.
    continues = tf.logical_not(tf.logical_or(terminateds, truncateds))
    # Probabilities that episode continues, computed by our continue predictor.
    # [B]
    continue_distr = forward_train_outs["continue_distribution"]
    # -log(p) loss
    continue_loss = - continue_distr.log_prob(continues)

    return decoder_loss + reward_loss + continue_loss


def world_model_dynamics_and_representation_loss(forward_train_outs):
    # Actual distribution over stochastic internal states (z) produced by the encoder.
    z_distr_encoder = forward_train_outs["z_distribution_encoder"]
    # Actual distribution over stochastic internal states (z) produced by the
    # dynamics network.
    z_distr_dynamics = forward_train_outs["z_distribution_dynamics"]

    # Stop gradient for encoder's z-outputs:
    sg_z_distr_encoder = tfp.distributions.Categorical(
        logits=tf.stop_gradient(z_distr_encoder.logits)
    )
    # Stop gradient for dynamics model's z-outputs:
    sg_z_distr_dynamics = tfp.distributions.Categorical(
        logits=tf.stop_gradient(z_distr_dynamics.logits)
    )

    # Implement free bits. According to [1]:
    # "To avoid a degenerate solution where the dynamics are trivial to predict but
    # contain not enough information about the inputs, we employ free bits by clipping
    # the dynamics and representation losses below the value of 1 nat ≈ 1.44 bits. This
    # disables them while they are already minimized well to focus the world model
    # on its prediction loss"
    L_dyn = tf.math.maximum(
        1.0,
        tfp.distributions.kl_divergence(sg_z_distr_encoder, z_distr_dynamics),
    )
    L_rep = tf.math.maximum(
        1.0,
        tfp.distributions.kl_divergence(z_distr_encoder, sg_z_distr_dynamics),
    )
    return L_dyn, L_rep
