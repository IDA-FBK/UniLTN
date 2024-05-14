import os
from collections import defaultdict

import tensorflow as tf

def create_dir(dir_path):
    """
    Create a directory if it does not exist.

    Parameters:
    - dir_path (str): The path of the directory to be created.

    Returns:
    - None
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


def get_scheduled_parameters_mnist_addition(epochs):
    """
    Generate scheduled parameters for mnist addition tasks.

    Parameters:
    - epochs (int): Total number of epochs.

    Returns:
    - dict: Dictionary containing scheduled parameters for each epoch.
    """
    scheduled_parameters = defaultdict(lambda: {})
    for epoch in range(0, 4):
        scheduled_parameters[epoch] = {"p_schedule": tf.constant(1.)}
    for epoch in range(4, 8):
        scheduled_parameters[epoch] = {"p_schedule": tf.constant(2.)}
    for epoch in range(8, 12):
        scheduled_parameters[epoch] = {"p_schedule": tf.constant(4.)}
    for epoch in range(12, epochs):
        scheduled_parameters[epoch] = {"p_schedule": tf.constant(6.)}

    return scheduled_parameters

