"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""

import tensorflow as tf


@tf.function
def disagree_loss(dream_data):
    z_predicted_probs_N_HxB = dream_data["z_predicted_probs_N_HxB"]

    # Targets are the next step z-states.
    targets_t1_to_H_B = tf.stop_gradient(dream_data["z_states_prior_t0_to_H_B"])[1:]

    shape = tf.shape(targets_t1_to_H_B)
    H, B = shape[0], shape[1]
    # Fold z-dimensions (num categoricals x num classes)
    targets_t1_to_H_B = tf.reshape(targets_t1_to_H_B, shape=[H, B, -1])

    loss = 0.0
    for net_idx in range(len(z_predicted_probs_N_HxB)):
        z_predicted_HxB = z_predicted_probs_N_HxB[net_idx]
        # Unfold time (H) rank and fold z-dims (num_categoricals x num_classes).
        z_predicted_t0_to_H_B = tf.reshape(
            z_predicted_HxB,
            shape=[H+1, B, -1],
        )
        del z_predicted_HxB
        z_predicted_t0_to_Hm1_B = z_predicted_t0_to_H_B[:-1]
        del z_predicted_t0_to_H_B
        # MSE diff between predicted next z-states (from the disagree network) and
        # actual dreamed next z-states (by the dynamics (prior) net).
        # Mask out everything on or past a dreamed continue=False flag.
        mse_H_B = tf.reduce_sum(
            tf.math.square(z_predicted_t0_to_Hm1_B - targets_t1_to_H_B),
            axis=-1,
        )# TEST: no weights, like in Danijar's code: * tf.stop_gradient(dream_data["dream_loss_weights_t0_to_H_B"])[1:]
        loss += tf.reduce_mean(mse_H_B)
    return loss
