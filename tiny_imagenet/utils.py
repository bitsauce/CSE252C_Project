import os
import numpy as np
import tensorflow as tf

def get_log_dir(model_name):
    """
    Returns the logging directory given the model name

    Args:
        model_name: Name of the model

    Returns:
        Path to a unique logging directory for this run
    """

    run_idx = 0
    while True:
        log_dir = os.path.join(model_name, "run_{}".format(run_idx))
        if not os.path.isdir(log_dir):
            return log_dir
        run_idx += 1

def add_scalar_summary(name, value):
    return tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value),])
