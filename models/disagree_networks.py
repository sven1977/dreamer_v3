"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""

import tensorflow as tf

from models.components.mlp import MLP
from models.components.representation_layer import RepresentationLayer


class DisagreeNetworks(tf.keras.Model):
    """Predict the RSSM's z^(t+1), given h(t), z^(t), and a(t).

    Disagreement (stddev) between the N networks in this model on what the next z^ would
    be are used to produce intrinsic rewards for enhanced, curiosity-based exploration.

    TODO
    """
    def __init__(self, *, num_networks, model_dimension, intrinsic_rewards_scale):
        super().__init__()

        self.model_dimension = model_dimension
        self.num_networks = num_networks
        self.intrinsic_rewards_scale = intrinsic_rewards_scale

        self.mlps = []
        self.representation_layers = []

        for _ in range(self.num_networks):
            self.mlps.append(MLP(
                model_dimension=model_dimension,
                output_layer_size=None,
                trainable=True,
            ))
            self.representation_layers.append(RepresentationLayer())

        # Optimizer.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-5)

    def call(self, inputs, z, a_one_hot, training=None):
        return self.forward_train(h=inputs, z=z, a_one_hot=a_one_hot)

    def compute_intrinsic_rewards(self, dream_data, forward_train_out):
        shape = tf.shape(dream_data["h_states_t0_to_H_B"])
        Hp1, B = shape[0], shape[1]

        # Intrinsic rewards are computed as:
        # Stddev of the mode of the 32x32 discrete, stochastic probs
        # between the different nets. Meaning that if the larger the disagreement
        # (stddev) between the nets on what the probabilities for the different
        # classes should be, the higher the intrinsic reward.
        z_predicted_probs_N_HxB = forward_train_out["z_predicted_probs_N_HxB"]
        N = len(z_predicted_probs_N_HxB)
        z_predicted_modes_N_HxB = tf.stack(z_predicted_probs_N_HxB, axis=0)
        # Flatten z-dims (num_categoricals x num_classes).
        z_predicted_modes_N_HxB = tf.reshape(
            z_predicted_modes_N_HxB, shape=(N, Hp1*B, -1)
        )

        # Compute stddevs over all disagree nets.
        # Mean over last axis ([num categoricals] x [num classes] folded axis).
        # Unfold time axis.
        stddevs_H_B_mean = tf.reshape(
            tf.reduce_mean(
                tf.math.reduce_std(z_predicted_modes_N_HxB, axis=0),
                axis=-1,
            ),
            shape=(Hp1, B),
        )
        return stddevs_H_B_mean * self.intrinsic_rewards_scale

    def forward_train(self, h, z, a_one_hot):
        HxB = tf.shape(h)[0]
        z = tf.reshape(z, shape=(HxB, -1))
        inputs_ = tf.stop_gradient(tf.concat([h, z, a_one_hot], axis=-1))

        z_predicted_probs_N_HxB = [
            repr(mlp(inputs_), return_z_probs=True)[1]  # [0]=sample; [1]=returned probs
            for mlp, repr in zip(self.mlps, self.representation_layers)
        ]
        # shape=(N, HxB, [num categoricals], [num classes]); N=number of disagree nets.
        # HxB -> folded horizon_H x batch_size_B (from dreamed data).

        return {"z_predicted_probs_N_HxB": z_predicted_probs_N_HxB}
