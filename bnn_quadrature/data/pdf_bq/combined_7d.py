from typing import Union

import numpy as np

from bnn_quadrature.extend_emukit.truncated_gaussian_measure import TruncatedGaussianMeasure


def combined_7d(a: Union[float, None], b: Union[float, None]):
    weights = [20, 25, 9]
    return [
               TruncatedGaussianMeasure(
                   mean=np.array([1.33, 0.38, 4e-3, 1.33, 0.0, 100., 100.]),
                   variance=np.array([0.1, 0.001, 1e-8, 0.1, 50.0, 0.5, 0.1]),
                   a=np.array([None, None, None, None, a, None, None]),
                   b=np.array([None, None, None, None, b, None, None]),
               ),
               TruncatedGaussianMeasure(
                   mean=np.array([1.33, 0.38, 4e-3, 1.33, 22.5, 100., 100.]),
                   variance=np.array([0.1, 0.001, 1e-8, 0.1, 40.0, 0.5, 0.1]),
                   a=np.array([None, None, None, None, a, None, None]),
                   b=np.array([None, None, None, None, b, None, None]),
               ),
               TruncatedGaussianMeasure(
                   mean=np.array([1.33, 0.38, 4e-3, 1.33, 33.75, 100., 100.]),
                   variance=np.array([0.1, 0.001, 1e-8, 0.1, 8.0, 0.5, 0.1]),
                   a=np.array([None, None, None, None, a, None, None]),
                   b=np.array([None, None, None, None, b, None, None]),
               ),
           ], weights / np.sum(weights)
