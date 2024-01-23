import warnings
from typing import List, Tuple, ContextManager, Union

import numpy as np
from emukit.quadrature.kernels.integration_measures import IntegrationMeasure

"""
The following function is adapted from Emukit 0.4.9
( https://github.com/EmuKit/emukit/releases/tag/0.4.9
Copyright (c) the Apache 2.0,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to produce a truncated Gaussian Measure.
"""

class TruncatedGaussianMeasure(IntegrationMeasure):
    """
    The isotropic Gaussian measure.

    An isotropic Gaussian is a Gaussian with scalar co-variance matrix. The density is
    :math:`p(x)=(2\pi\sigma^2)^{-\frac{D}{2}} e^{-\frac{1}{2}\frac{\|x-\mu\|^2}{\sigma^2}}`
    """

    def __init__(
        self,
        mean: np.ndarray,
        variance: Union[float, np.ndarray],
        a: Union[float, None, np.ndarray],
        b: Union[float, None, np.ndarray],
    ):
        """
        :param mean: the mean of the Gaussian, shape (num_dimensions, )
        :param variance: the scalar variance of the isotropic covariance matrix of the Gaussian.
        """
        super().__init__("GaussianMeasure")
        self._check_inputs(mean, variance, a, b)

        self.mean = mean
        self.variance = variance
        self.a = a
        self.b = b
        self.num_dimensions = mean.shape[0]

    @staticmethod
    def _check_inputs(mean, variance, a, b):
        if not isinstance(mean, np.ndarray):
            raise TypeError(
                "Mean must be of type numpy.ndarray, {} given.".format(type(mean))
            )
        if mean.shape[-1] > 1:
            if not isinstance(variance, np.ndarray):
                raise TypeError(
                    "For d>1 variance must be of type numpy.ndarray, {} given.".format(type(mean))
                )
            if not isinstance(a, np.ndarray):
                raise TypeError(
                    "For d>1 a must be of type numpy.ndarray, {} given.".format(type(mean))
                )
            if not isinstance(b, np.ndarray):
                raise TypeError(
                    "For d>1 b must be of type numpy.ndarray, {} given.".format(type(mean))
                )
            if not (variance > 0).all():
                raise ValueError(
                    "Variance must be positive, current value is {}.".format(variance)
                )
        else:
            if not variance > 0:
                raise ValueError(
                    "Variance must be positive, current value is {}.".format(variance)
                )


    @property
    def full_covariance_matrix(self):
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        return self.variance * np.eye(self.num_dimensions)

    @property
    def can_sample(self) -> bool:
        """
        Indicates whether probability measure has sampling available.
        :return: True if sampling is available
        """
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        return True

    def compute_density(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the density at point x
        :param x: points at which density is computed, shape (num_points, num_dimensions)
        :return: the density at x, shape (num_points, )
        """
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        factor = (2 * np.pi * self.variance) ** (self.num_dimensions / 2)
        scaled_diff = (x - self.mean) / (np.sqrt(2 * self.variance))
        return np.exp(-np.sum(scaled_diff**2, axis=1)) / factor

    def compute_density_gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the density at point x
        :param x: points at which the gradient is computed, shape (num_points, num_dimensions)
        :return: the gradient of the density at x, shape (num_points, num_dimensions)
        """
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        values = self.compute_density(x)
        return ((-values / self.variance) * (x - self.mean).T).T

    def get_box(self) -> List[Tuple[float, float]]:
        """
        Meaningful box-bounds around the measure. Outside this box, the measure should be virtually zero.

        :return: box in which the measure lies. List of D tuples, where D is the dimensionality and the tuples contain
        the lower and upper bounds of the box i.e., [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)]
        """
        # Note: the factor 10 is somewhat arbitrary but well motivated. If this method is used to get a box for
        # data-collection, the box will be 2x 10 standard deviations wide in all directions, centered around the mean.
        # Outside the box the density is virtually zero.
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        factor = 10
        lower = self.mean - factor * np.sqrt(self.variance)
        upper = self.mean + factor * np.sqrt(self.variance)
        return list(zip(lower, upper))

    def get_samples(
        self, num_samples: int, context_manager: ContextManager = None
    ) -> np.ndarray:
        """
        Samples from the isotropic Gaussian distribution.

        :param num_samples: number of samples
        :param context_manager: The context manager that contains variables to fix and the values to fix them to. If a
        context is given, this method samples from the conditional distribution.
        :return: samples, shape (num_samples, num_dimensions)
        """
        warnings.warn("TODO! Not implemented!!! Uses implementation from Gaussian")
        samples = self.mean + np.sqrt(self.variance) * np.random.randn(
            num_samples, self.num_dimensions
        )

        if context_manager is not None:
            # since the Gaussian is isotropic, fixing the value after sampling the joint is equal to sampling the
            # conditional.
            samples[:, context_manager.context_idxs] = context_manager.context_values

        return samples
