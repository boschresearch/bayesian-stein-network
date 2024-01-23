from typing import Union

import torch

from bnn_quadrature.data.dataset_base import GenericDataClass
from bnn_quadrature.data.pdf.wind_farm_pdf import Combination7dPDF
from bnn_quadrature.data.pdf_bq.combined_7d import combined_7d
from bnn_quadrature.data.wind_farm.extended_wind_farm_model import WindData


class WindFarmDataset7D(GenericDataClass):
    NAME = "wind_farm_7d"
    dim = 7

    def __init__(
        self,
        version: int,
        use_x_rescaling: bool = False,
        use_y_rescaling: bool = False,
        *args,
        **kwargs
    ):
        super(WindFarmDataset7D, self).__init__(
            use_x_rescaling, use_y_rescaling, args, kwargs
        )
        self.name = f"{self.NAME}_v_{version}"
        self.data = WindData()
        self.a = 0.
        self.b = 45.
        self.pdf = Combination7dPDF()
        self.emukit_pdf = combined_7d(self.a, self.b)
        self.index_of_bounds = torch.tensor([4])

    def sample(self):
        ct_prime, wake1, wake2, ti, theta, hub_height, diameter = torch.chunk(
            self.pdf.sample(1), self.dim, dim=-1
        )
        self.data.ct_prime = ct_prime.item()
        self.data.wake1 = wake1.item()
        self.data.wake2 = wake2.item()
        self.data.turbulence_intensity = ti.item()
        self.data.theta = theta.item()
        self.data.hub_height = hub_height.item()
        self.data.diameter = diameter.item()

    def f(self, x: torch.Tensor) -> Union[None, torch.Tensor]:
        return None

    def true_solution(self):
        return None






