import numpy as np
import torch
from torch.distributions import MultivariateNormal

from bnn_quadrature.data.pdf.pdf import (
    PDF,
    NormalPDF,
)


class MixtureNormalPDF2(NormalPDF):
    """
    Gaussian probability density function with mean 0 and variance 1
    """

    def __init__(
        self,
    ):
        super(MixtureNormalPDF2, self).__init__()
        self.in_dim = 1
        self.param_list = [
            {
                "loc": torch.ones(self.in_dim) * 0,
                "covariance_matrix": torch.eye(self.in_dim) * 50.0,
            },
            {
                "loc": torch.ones(self.in_dim) * 22.5,
                "covariance_matrix": torch.eye(self.in_dim) * 40.0,
            },
            {
                "loc": torch.ones(self.in_dim) * 33.75,
                "covariance_matrix": torch.eye(self.in_dim) * 8.0,
            },
        ]
        self.weights = torch.tensor([20, 25, 9])
        self.weights = self.weights / torch.sum(self.weights)

    def _apply(self, fn):
        super(NormalPDF, self)._apply(fn)
        for param_dict in self.param_list:
            for key, v in param_dict.items():
                param_dict[key] = fn(v)
        self.weights = fn(self.weights)

    def sample(self, dataset_size: int):
        params = np.random.choice(
            a=self.param_list, p=self.weights.numpy(), size=dataset_size
        )
        out = []
        for param in params:
            out.append(MultivariateNormal(**param).sample((1,)))
        out = torch.stack(out)
        return out

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x:
        :return:
        """
        dist_list = []
        for params in self.param_list:
            dist_list.append(MultivariateNormal(**params))
        sol_list = []
        for d in dist_list:
            sol_list.append(torch.exp(d.log_prob(x)))
        sol_list = torch.stack(sol_list, 0)
        sol_list = sol_list * self.weights.unsqueeze(1).unsqueeze(-1)
        sol = torch.sum(sol_list, dim=0)
        return sol


class Combination7dPDF(PDF):
    def __init__(self):
        super(Combination7dPDF, self).__init__()
        self.in_dim = 5
        pdf_ct_prime = NormalPDF(mean=1.33, var=0.1)
        pdf_wake1 = NormalPDF(mean=0.38, var=0.001)
        pdf_wake2 = NormalPDF(mean=4e-3, var=1e-8)
        pdf_turbulence_intensity = NormalPDF(mean=1.33, var=0.1)
        pdf_theta = MixtureNormalPDF2()
        pdf_hub_height = NormalPDF(mean=100, var=0.5)
        pdf_diameter = NormalPDF(mean=100, var=0.1)
        self.pdf_list = [
            pdf_ct_prime,
            pdf_wake1,
            pdf_wake2,
            pdf_turbulence_intensity,
            pdf_theta,
            pdf_hub_height,
            pdf_diameter
        ]
        self.num_variables = len(self.pdf_list)

    def _apply(self, fn):
        super(Combination7dPDF, self)._apply(fn)
        for i, pdf in enumerate(self.pdf_list):
            pdf._apply(fn)
            self.pdf_list[i] = pdf

    def pdf(self, x: torch.Tensor):
        variables = torch.chunk(x, self.num_variables, dim=-1)
        prod_pdf = 1.
        for var, pdf in zip(variables, self.pdf_list):
            prod_pdf *= pdf(var)
        return prod_pdf

    def sample(self, dataset_size):
        x = []
        for pdf in self.pdf_list:
            x.append(pdf.sample(dataset_size))
        x = torch.stack(x, dim=-1)
        # assert x.shape == [dataset_size, 1, len(self.pdf_list)]
        return x
