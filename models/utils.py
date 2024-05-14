import tensorflow as tf


def get_optimizer(conf):
    """
    Get an optimizer based on the provided configuration.

    Parameters:
    - conf (dict): Configuration for the optimizer, including the optimizer type and learning rate.

    Returns:
    - tf.keras.optimizers.Optimizer or None: The optimizer instance based on the configuration, or None if the optimizer type is not supported.
    """
    optimizer = None
    if conf["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=conf["learning_rate"])
    elif conf["optimizer"] == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=conf["learning_rate"])

    return optimizer
