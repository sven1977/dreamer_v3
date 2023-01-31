"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf


def two_hot(value, num_buckets, lower_bound=20.0, upper_bound=20.0):
    """Returns a two-hot vector of dim=num_buckets with two entries != 0.0.

    Entries in the vector represent equally sized buckets within some fixed range
    (`lower_bound` to `upper_bound`).
    Those entries not 0.0 at positions k and k+1 encode the actual `value` and sum
    up to 1.0. They are the weights multiplied by the buckets values at k and k+1 for
    retrieving `value`.

    Example:
        num_buckets=11
        lower_bound=-5
        upper_bound=5
        value=2.5
        -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]
        -> [-5   -4   -3   -2   -1   0    1    2    3    4    5] (0.5*2 + 0.5*3=2.5)

    Example:
        num_buckets=5
        lower_bound=-1
        upper_bound=1
        value=0.1
        -> [0.0, 0.0, 0.8, 0.2, 0.0]
        -> [-1  -0.5   0   0.5   1] (0.2*0.5 + 0.8*0=0.1)
    """
    # Tensor of batch indices: [0, B=batch size).
    batch_indices = tf.range(0, value.shape[0], dtype=tf.float32)
    # Calculate the step deltas (how much space between each bucket's central value?).
    delta = (upper_bound - lower_bound) / (num_buckets - 1)
    # Compute the float indices (might be non-int numbers: sitting between two buckets).
    idx = (-lower_bound + value) / delta
    # k
    k = tf.math.floor(idx)
    # k+1
    kp1 = tf.math.ceil(idx)
    # The actual values found at k and k+1 inside the set of buckets.
    values_k = lower_bound + k * delta
    values_kp1 = lower_bound + kp1 * delta
    # Compute the two-hot weights (adding up to 1.0) to use at index k and k+1.
    weights_k = (value - values_kp1) / (values_k - values_kp1)
    weights_kp1 = 1.0 - weights_k
    # Compile a tensor of full paths (indices from batch index to feature index) to
    # use for the scatter_nd op.
    indices_k = tf.stack([batch_indices, k], -1)
    indices_kp1 = tf.stack([batch_indices, kp1], -1)
    indices = tf.concat([indices_k, indices_kp1], 0)
    # The actual values (weights adding up to 1.0) to place at the computed indices.
    updates = tf.concat([weights_k, weights_kp1], 0)
    # Call the actual scatter update op, returning a zero-filled tensor, only changed
    # at the given indices.
    return tf.scatter_nd(
        tf.cast(indices, tf.int32),
        updates,
        shape=(value.shape[0], num_buckets),
    )


if __name__ == "__main__":
    print(two_hot(tf.convert_to_tensor([2.5, 0.1]), 11, -5.0, 5.0))
    print(two_hot(tf.convert_to_tensor([0.1]), 5, -1.0, 1.0))
    print(two_hot(tf.convert_to_tensor([-0.5, -1.2]), 10, -6.0, 3.0))
