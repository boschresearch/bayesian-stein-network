
import gpytorch
import torch
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ZeroMean

from bnn_quadrature.models.gp.stein_kernel import  SteinKernelRBF


class SteinGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Likelihood,
        kernel: SteinKernelRBF,
    ):
        super(SteinGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.sk = kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.sk(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModel(gpytorch.models.ExactGP):
    """
    Implements Stein Kernel with prior on theta ~ N(0, infty). It is not possible to use
    hyperparameter opimization with this kernel
    """

    def __init__(
        self,
        gp: SteinGPModel,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: Likelihood,
    ):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.gp = gp
        self.likelihood = likelihood
        self.x = train_x
        self.y = train_y

    def forward(self, *inputs, **kwargs):
        """
        Not implemented.
        :param inputs:
        :param kwargs:
        :return:
        """
        pass

    def mll(self, x, y):
        K = self.get_K(x)
        K_inv_y = torch.linalg.solve(K, y)
        y_K_inv_y = y.T @ K_inv_y
        logdet_K = torch.logdet(K)
        return -0.5 * y_K_inv_y - 0.5 * logdet_K

    def get_theta_0(self) -> torch.Tensor:
        """
        Computes posterior for theta
        x: Training data
        :return:
        """
        K_inv_h = self.get_K_inv_h(self.x)
        theta_var = self.get_theta_0_var(self.x, self.y)
        theta = theta_var @ (K_inv_h.T @ self.y)
        return theta

    def get_theta_0_var(self, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):
        K_inv_h = self.get_K_inv_h(x)
        h = torch.ones((1, x.shape[0]))
        theta_var_inv = h @ K_inv_h
        self.theta_var = 1 / theta_var_inv
        return self.theta_var

    def get_K(self, x):
        K = self.gp.sk(x, x).evaluate()
        K = K + torch.eye(K.shape[0]) * self.likelihood.noise
        self.K = K
        return self.K

    def get_K_inv_h(self, x):
        K = self.get_K(x)
        h = torch.ones((1, x.shape[0]))
        K_inv_h = torch.linalg.solve(K, h.T)
        self.K_inv_h = K_inv_h
        return self.K_inv_h

