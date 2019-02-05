import tensorflow as tf


def weibull_beta_penalty(b_, location=10.0, growth=20.0, name=None):
    # Regularization term to keep beta below location

    with tf.name_scope(name):
        scale = growth / location
        penalty_ = tf.exp(scale * (b_ - location))

    return (penalty_)

