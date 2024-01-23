import math
import warnings
from typing import Union

import numpy as np
import scipy.special
from emukit.quadrature.interfaces import IRBF
from emukit.quadrature.kernels.quadrature_rbf import QuadratureRBF

from bnn_quadrature.extend_emukit.truncated_gaussian_measure import TruncatedGaussianMeasure

"""
The following function is adapted from Emukit 0.4.9
( https://github.com/EmuKit/emukit/releases/tag/0.4.9
Copyright (c) the Apache 2.0,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to produce a truncated Gaussian Measure.
"""


def normalization_trunc_gauss(
    a: Union[float, None], b: Union[float, None], mean: np.ndarray, sigma: float
) -> np.ndarray:
    """
    Computes \int_a^b N(mean, sigma**2)
    :param a:
    :param b:
    :param mean:
    :param sigma:
    :return:
    """
    if a is None and b is None:
        return np.ones_like(mean)
    if b is None:
        return 1 - cum_distribution((a - mean) / sigma)
    if a is None:
        return cum_distribution((b - mean) / sigma)
    return cum_distribution((b - mean) / sigma) - cum_distribution((a - mean) / sigma)


def cum_distribution(x: np.ndarray) -> np.ndarray:
    return 1.0 / 2.0 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))


class QuadratureRBFTruncatedGaussian(QuadratureRBF):
    """
    Augments an RBF kernel with integrability

    Note that each standard kernel goes with a corresponding quadrature kernel, in this case standard rbf kernel.
    """

    def __init__(
        self,
        rbf_kernel: IRBF,
        measure: TruncatedGaussianMeasure,
        variable_names: str = "",
    ) -> None:
        """
        :param rbf_kernel: standard emukit rbf-kernel
        :param measure: a Gaussian measure
        :param variable_names: the (variable) name(s) of the genz_data
        """
        super().__init__(
            rbf_kernel=rbf_kernel,
            integral_bounds=None,
            measure=measure,
            variable_names=variable_names,
        )

    def qK(self, x2: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """
        RBF kernel with the first component integrated out aka kernel mean

        :param x2: remaining argument of the once integrated kernel, shape (n_point N, input_dim)
        :param scale_factor: scales the lengthscale of the RBF kernel with the multiplicative factor.
        :returns: kernel mean at location x2, shape (1, N)
        """
        dim = self.measure.mean.shape[-1]
        lengthscale = scale_factor * self.lengthscale

        if dim > 1:
            kernel_mean = np.ones((x2.shape[0], 1))
            for i, (a, b, mean1, var1) in enumerate(zip(
                self.measure.a, self.measure.b, self.measure.mean, self.measure.variance
            )):
                sigma1 = np.sqrt(var1)
                kernel_mean *= self.compute_kernel_mean(
                    a, b, lengthscale, mean1, sigma1, x2[:, i][:, None]
                )
        else:
            kernel_mean = self.compute_kernel_mean(
                self.measure.a,
                self.measure.b,
                lengthscale,
                self.measure.mean,
                np.sqrt(self.measure.variance),
                x2,
            )
        return kernel_mean.reshape(1, -1)*self.variance

    def compute_kernel_mean(self, a, b, lengthscale, mean1, sigma1, x2):
        z1 = normalization_trunc_gauss(a=a, b=b, mean=mean1, sigma=sigma1)
        sigma_sum = sigma1**2 + lengthscale**2
        product_mean = (sigma1**2 * x2 + lengthscale**2 * mean1) / sigma_sum
        product_sigma = np.sqrt(sigma1**2 * lengthscale**2 / sigma_sum)
        scale = (
            1
            / np.sqrt(2 * math.pi * sigma_sum)
            * np.exp(-((mean1 - x2) ** 2) / (2 * sigma_sum))
        )
        z2 = normalization_trunc_gauss(
            a=a, b=b, mean=product_mean, sigma=product_sigma
        )
        z_ = scale * np.sqrt(2 * math.pi) * lengthscale * z2
        kernel_mean = z_ / z1
        return kernel_mean

    def qKq(self) -> float:
        """
        RBF kernel integrated over both arguments x1 and x2
        Only works for infinite integrals

        :returns: double integrated kernel
        """
        warnings.warn("Implementation only valid for Gaussian measures, if both a and b are None this is true!")
        dim = self.measure.mean.shape[-1]
        lengthscale = self.lengthscale
        if dim > 1:
            kernel_var = 1.
            for i, var in enumerate(self.measure.variance):
                kernel_var *= self.compute_kernel_covariance(var, lengthscale)
        else:
            var = self.measure.variance
            kernel_var = self.compute_kernel_covariance(var, lengthscale)
        return kernel_var

    def compute_kernel_covariance(self, var, l):
        """
        RBF kernel integrated over both arguments for one dimension
        :param var: Variance of the PDF
        :param l: lengthscale of the kernel
        :return: float,
        """
        return 1/np.sqrt(1 + 2 * var/l**2)

    def dqK_dx(self, x2: np.ndarray) -> np.ndarray:
        """
        gradient of the kernel mean (integrated in first argument) evaluated at x2

        :param x2: points at which to evaluate, shape (n_point N, input_dim)
        :return: the gradient with shape (input_dim, N)
        """
        qK_x = self.qK(x2)
        factor = 1.0 / (self.lengthscale**2 + self.measure.variance)
        return -(qK_x * factor) * (x2 - self.measure.mean).T
