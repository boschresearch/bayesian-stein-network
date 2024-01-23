import numpy as np

from bnn_quadrature.extend_emukit.truncated_gaussian_measure import (
    TruncatedGaussianMeasure,
)


def standard_normal(dim=1):
    if dim > 1:
        a = np.array([None] * dim)
        b = np.array([None] * dim)
        return [
            TruncatedGaussianMeasure(
                mean=np.zeros(dim), variance=np.ones(dim), a=a, b=b
            )
        ], [1.0]
    else:
        return [
            TruncatedGaussianMeasure(
                mean=np.zeros(dim), variance=1.0, a=None, b=None
            )
        ], [1.0]
