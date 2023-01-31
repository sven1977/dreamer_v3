"""
[1] Mastering Diverse Domains through World Models - 2023
D. Hafner, J. Pasukonis, J. Ba, T. Lillicrap
https://arxiv.org/pdf/2301.04104v1.pdf
"""
import tensorflow as tf


def symlog(x):
    """The symlog function as described in [1]"""
    return tf.math.sign(x) * tf.math.log(tf.math.abs(x) + 1)


def inverse_symlog(y):
    # To get to symlog inverse, we solve the symlog equation for x:
    #     y = sign(x) * log(|x| + 1)
    # <=> y / sign(x) = log(|x| + 1)
    # <=> y =  log( x + 1) V x >= 0
    #    -y =  log(-x + 1) V x <  0
    # <=> exp(y)  =  x + 1  V x >= 0
    #     exp(-y) = -x + 1  V x <  0
    # <=> exp(y)  - 1 =  x   V x >= 0
    #     exp(-y) - 1 = -x   V x <  0
    # <=>  exp(y)  - 1 = x   V x >= 0 (if x >= 0, then y must also be >= 0)
    #     -exp(-y) - 1 = x   V x <  0 (if x < 0, then y must also be < 0)
    # <=> sign(y) * (exp(|y|) - 1) = x
    return tf.math.sign(y) * (tf.math.exp(tf.math.abs(y)) - 1)
