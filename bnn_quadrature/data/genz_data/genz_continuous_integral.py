import numpy as np
import torch
from pydantic import BaseModel
from scipy.stats import qmc

from bnn_quadrature.data.dataset_base import DataClass
from bnn_quadrature.data.genz_data.genz_functions import (
    Genz_continuous,
    integral_Genz_continuous,
    uniform_to_gaussian,
)
from bnn_quadrature.data.pdf.pdf import StandardNormalPDF
from bnn_quadrature.data.pdf_bq.gaussian import standard_normal


class ContinuousGenzOptions(BaseModel):
    """
    These options could be extended to np.Arrays and so on...
    """

    a1: float = 1.3
    a2: float = .55  # 0.5 means centered at 0!


class ContinuousGenz(DataClass):
    NAME = "genz"

    def __init__(
        self,
        use_x_rescaling: bool,
        use_y_rescaling: bool,
        dataset_size: int,
        version: int,
        dim: int = 1,
    ):
        super(ContinuousGenz, self).__init__(
            use_x_rescaling, use_y_rescaling, dataset_size, dim=dim, version=version
        )
        opts = ContinuousGenzOptions()
        self.dim = dim
        self.a1 = np.repeat(opts.a1, dim)
        self.a2 = np.repeat(opts.a2, dim)
        self.base_size = int(1e7)
        self.pdf = StandardNormalPDF(dim=dim)
        self.emukit_pdf = standard_normal(dim=dim)
        self.name = f"{self.NAME}_dim_{self.dim}_v_{self.version}"

    def f(self, x: torch.Tensor):
        return torch.tensor(
            uniform_to_gaussian(self.uniform_genz)(x.squeeze(1).cpu().detach().numpy()),
            dtype=torch.float32,
        )

    def uniform_genz(self, x):
        return Genz_continuous(x, self.a1, self.a2)

    def true_solution(self) -> float:
        return float(integral_Genz_continuous(self.a1, self.a2))


class GenzContinuousQMCIntegral(ContinuousGenz):
    NAME = "genz_qmc"

    def generate_data(self):
        np.random.seed(self.version)
        torch.manual_seed(self.version)
        dist = qmc.MultivariateNormalQMC(mean=np.zeros(self.dim))
        sample = dist.random(self.n_max)
        x = torch.tensor(sample, dtype=torch.float32).unsqueeze(1)
        y = self.f(x)
        np.random.seed(0)
        torch.manual_seed(0)
        return y, x
