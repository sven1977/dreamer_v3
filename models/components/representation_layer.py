"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf

[2] Mastering Atari with Discrete World Models - 2021
D. Hafner, T. Lillicrap, M. Norouzi, J. Ba
https://arxiv.org/pdf/2010.02193.pdf
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class RepresentationLayer(tf.keras.layers.Layer):
    """A representation (z) generating layer.

    The value for z is the result of sampling from a categorical distribution with
    shape B x `num_classes`.
    """
    def __init__(
        self,
        num_categoricals: int = 32,
        num_classes_per_categorical: int = 32,
    ):
        super().__init__()

        self.num_categoricals = num_categoricals
        self.num_classes_per_categorical = num_classes_per_categorical

        self.z_generating_layer = tf.keras.layers.Dense(
            self.num_categoricals * self.num_classes_per_categorical,
            activation=None,
        )

    def call(self, input_):
        """Produces a discrete, differentiable z-sample from the output of our layer.

        First concatenates the incoming tensors (x from the observation-encoder, h from
        the sequence model).

        Pushes the concatenated vector through our dense layer, which outputs
        32(B=num categoricals)*32(c=num classes) logits. Logits are used to:

        1) sample stochastically
        2) compute probs
        3) make sure sampling step is differentiable (see [2] Algorithm 1):
            sample=one_hot(draw(logits))
            probs=softmax(logits)
            sample=sample + probs - stop_grad(probs)
            -> Now sample has the gradients of the probs.

        Args:
            input_: The input to our z-generating layer.
        """
        logits = self.z_generating_layer(input_)
        logits = tf.reshape(
            logits,
            shape=(-1, self.num_categoricals, self.num_classes_per_categorical),
        )
        distribution = tfp.distributions.Categorical(logits=logits)
        sample = tf.one_hot(
            distribution.sample(),
            depth=self.num_classes_per_categorical,
        )
        probs = tf.nn.softmax(logits)
        return sample + probs - tf.stop_gradient(probs)


if __name__ == "__main__":
    layer = RepresentationLayer(num_categoricals=32, num_classes_per_categorical=32)
    # encoder output
    x = np.random.random(size=(1, 128))
    # GRU output
    h = np.random.random(size=(1, 512))
    out = layer(tf.concat([x, h], axis=-1))
    print(out.shape)
