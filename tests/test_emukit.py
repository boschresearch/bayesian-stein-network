import GPy
import numpy as np
from emukit.model_wrappers import RBFGPy

from bnn_quadrature.extend_emukit.quadrature_rbf_truncated_gaussian import QuadratureRBFTruncatedGaussian
from bnn_quadrature.extend_emukit.truncated_gaussian_measure import TruncatedGaussianMeasure

x_init = np.linspace(0, 1, 15)[:, None]
y_init = x_init**2
gpy_model = GPy.models.GPRegression(
    X=x_init,
    Y=y_init,
    kernel=GPy.kern.RBF(input_dim=y_init.shape[1], lengthscale=0.5, variance=1.0),
)

emukit_rbf = RBFGPy(gpy_model.kern)


def test_BQ_truncated_gauss_qK_shape():
    N = 10
    x = np.linspace(0, 1, N)[:, None]
    measure = TruncatedGaussianMeasure(mean=np.array([0.]), variance=1., a=0., b=1.)
    bq = QuadratureRBFTruncatedGaussian(rbf_kernel=emukit_rbf, measure=measure)
    kernel_mean = bq.qK(x)
    assert kernel_mean.shape == (1, N)
