import numpy as np


def get_error(theta_array, true_value):
    se = np.abs(theta_array-true_value)/true_value
    error = np.mean(se, axis=0)
    std_error = np.std(se, axis=0)
    theta_mean = np.mean(theta_array, axis=0)

    return (
        theta_mean,
        error,
        std_error,
    )
