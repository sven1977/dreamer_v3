"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import numpy as np
import tensorflow as tf

from models.components.mlp import MLP
from models.components.representation_layer import RepresentationLayer


class DynamicsPredictor(tf.keras.Model):
    def __init__(
            self,
            num_categoricals: int = 32,
            num_classes_per_categorical: int = 32,
    ):
        super().__init__()

        self.mlp = MLP(output_layer_size=None)
        self.representation_layer = RepresentationLayer(
            num_categoricals=num_categoricals,
            num_classes_per_categorical=num_classes_per_categorical,
        )

    def call(self, h):
        """

        Args:
            h: The deterministic hidden state of the sequence model.
        """
        # Send internal state through MLP.
        out = self.mlp(h)
        # Generate a z vector (stochastic, discrete sample).
        z = self.representation_layer(out)
        return z


if __name__ == "__main__":
    # DreamerV2/3 Atari input space: B x 32 (num_categoricals) x 32 (num_classes)
    h_dim = 8
    inputs = np.random.random(size=(1, 8))
    model = DynamicsPredictor()
    out = model(inputs)
    print(out.shape)

