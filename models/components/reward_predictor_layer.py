"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf

from models.components.mlp import MLP
from utils.two_hot import two_hot


class RewardPredictorLayer(tf.keras.layers.Layer):
    """A layer outputting reward predictions using K bins and two-hot encoding.
    TODO:
    """
    def __init__(self, num_buckets: int = 255):
        super().__init__()
        self.num_buckets = num_buckets



    def call(self, h, z):
        """Computes a predicted reward using the two-hot encoded
        """
        pass



if __name__ == "__main__":
    h_dim = 8
    h = np.random.random(size=(1, 8))
    z = np.random.random(size=(1, 8, 8))

    model = RewardPredictorLayer(num_buckets=255)

    out = model()
    print(out.shape)
