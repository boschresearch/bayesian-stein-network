import numpy as np
import torch

from bnn_quadrature.data.dataset_base import DataClass
from bnn_quadrature.data.genz_data.genz_functions import uniform_to_gaussian, integral_Genz_cornerpeak, \
    Genz_cornerpeak, Genz_discontinuous, integral_Genz_discontinuous, Genz_gaussian, integral_Genz_gaussian, \
    Genz_oscillatory, integral_Genz_oscillatory, Genz_productpeak, integral_Genz_productpeak


def return_a_u(dim):
    a = np.repeat(5.0, dim)
    u = np.repeat(0.5, dim)
    return a, u


class GenzCornerPeak(DataClass):
    NAME = "genz_corner"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def genz(self, x):
        a, u = return_a_u(self.dim)
        return Genz_cornerpeak(x, a, u)

    def true_solution(self) -> float:
        a, u = return_a_u(self.dim)
        return float(integral_Genz_cornerpeak(a, u))


class DiscontinuousGenz(DataClass):
    NAME = "genz_discontinuous"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def genz(self, x):
        a, u = return_a_u(self.dim)
        return Genz_discontinuous(x, a, u)

    def true_solution(self) -> float:
        a, u = return_a_u(self.dim)
        return float(integral_Genz_discontinuous(a, u))


class GenzGaussian(DataClass):
    NAME = "genz_gaussian"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def genz(self, x):
        a, u = return_a_u(self.dim)
        return Genz_gaussian(x, a, u)

    def true_solution(self) -> float:
        a, u = return_a_u(self.dim)
        return float(integral_Genz_gaussian(a, u))


class GenzOscillatory(DataClass):
    NAME = "genz_oscillatory"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def genz(self, x):
        a, u = return_a_u(self.dim)
        return Genz_oscillatory(x, a, u)

    def true_solution(self) -> float:
        a, u = return_a_u(self.dim)
        return float(integral_Genz_oscillatory(a, u))


class GenzProductPeak(DataClass):
    NAME = "genz_product"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def genz(self, x):
        a, u = return_a_u(self.dim)
        return Genz_productpeak(x, a, u)

    def true_solution(self) -> float:
        a, u = return_a_u(self.dim)
        return float(integral_Genz_productpeak(a, u))
