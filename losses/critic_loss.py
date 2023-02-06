import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils.symlog import symlog


@tf.function
def critic_loss(
        dreamed_observations,
        dreamed_rewards,
        terminateds,
        truncateds,
        forward_train_outs,
):
    pass # TODO


def gae(rewards, continues, value_predictions, gamma, lambda_):
    pass #TODO
