import numpy
import numpy as np
import torch
from numpy.random import vonmises
from torch import nn
from torch.distributions import MultivariateNormal, LogNormal


class PDF(nn.Module):
    def __init__(self):
        super(PDF, self).__init__()
        self.in_dim = None
        self.mean = 0.
        self.std = 1.0

    def _apply(self, fn):
        super(PDF, self)._apply(fn)
        if isinstance(self.mean, torch.Tensor):
            self.mean = fn(self.mean)
        if isinstance(self.std, torch.Tensor):
            self.std = fn(self.std)

    def update_rescaling(self, mean, std):
        self.mean = mean
        self.std = std

    def grad_log(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor shape B x 1 x D
        :return: Tensor shape B x 1 x 1
        """
        v = torch.ones([x.shape[0], 1], device=x.device)
        with torch.enable_grad():
            x.requires_grad = True
            div_dlogp_dx = torch.autograd.grad(
                self.log_pdf(x),
                x,
                grad_outputs=v,
                create_graph=True,
            )[0]

        div_log_p = div_dlogp_dx
        return div_log_p

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Tensor shape B x 1 x D
        :return: Tensor shape B x 1
        """
        x = self.undo_rescale(x)
        return self.pdf(x)

    def undo_rescale(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.std + self.mean
        return x

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.forward(x))

    def sample(self, dataset_size: int) -> torch.Tensor:
        pass


class StandardNormalPDF(PDF):
    """
    Gaussian probability density function with mean 0 and variance 1
    """

    def __init__(self, dim: int = 1):
        super(StandardNormalPDF, self).__init__()
        self.in_dim = dim
        self.cov_matrix = torch.diag(torch.ones(self.in_dim)).unsqueeze(0)
        self.loc = torch.zeros(self.in_dim)
        self.dist = MultivariateNormal(loc=self.loc, precision_matrix=self.cov_matrix)

    def _apply(self, fn):
        super(StandardNormalPDF, self)._apply(fn)
        self.cov_matrix = fn(self.cov_matrix)
        self.loc = fn(self.loc)
        self.dist = MultivariateNormal(loc=self.loc, precision_matrix=self.cov_matrix)

    def pdf(self, x) -> torch.Tensor:
        out = torch.exp(self.dist.log_prob(x))
        return out

    def grad_log(self, x: torch.Tensor) -> torch.Tensor:
        return -self.undo_rescale(x)*self.std

    def sample(self, dataset_size: int) -> torch.Tensor:
        data = self.dist.rsample((dataset_size,))
        return data


class NormalPDF(PDF):
    """
    Gaussian probability density function with mean 0 and variance 1
    """

    def __init__(self, mean: float = 0.0, var: float = 1.0, dim: int = 1):
        super(NormalPDF, self).__init__()
        self.in_dim = dim
        self.cov_matrix = torch.diag(torch.ones(self.in_dim)).unsqueeze(0) / var
        self.loc = torch.ones(self.in_dim) * mean
        self.dist = MultivariateNormal(loc=self.loc, precision_matrix=self.cov_matrix)

    def _apply(self, fn):
        super(NormalPDF, self)._apply(fn)
        self.cov_matrix = fn(self.cov_matrix)
        self.loc = fn(self.loc)
        self.dist = MultivariateNormal(loc=self.loc, precision_matrix=self.cov_matrix)

    def pdf(self, x) -> torch.Tensor:
        out = torch.exp(self.dist.log_prob(x))
        return out

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        x = self.undo_rescale(x)
        return self.dist.log_prob(x)

    def sample(self, dataset_size: int) -> torch.Tensor:
        data = self.dist.rsample((dataset_size,))
        return data


