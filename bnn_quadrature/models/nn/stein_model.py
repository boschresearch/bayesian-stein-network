import os
from typing import Union

import torch
from torch import nn

from bnn_quadrature.data.pdf.pdf import PDF
from bnn_quadrature.evaluation.fit_laplace import fit_laplace
from bnn_quadrature.models.nn.stein_layer import SteinLayer, SteinScoreLayer
from bnn_quadrature.options.enums import TransformationEnum
from bnn_quadrature.options.options import Options


class SteinModel(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        pdf: PDF,
        start_theta_0: float,
        transformation: TransformationEnum,
        const: float = 1.,
        index_of_bounds: Union[torch.Tensor, None] = None,
        a: Union[torch.Tensor, float, None] = None,
        b: Union[torch.Tensor, float, None] = None,
    ):
        super(SteinModel, self).__init__()
        self.model = SteinLayer(
            network,
            pdf,
            start_theta_0,
            transformation,
            const=const,
            a=a,
            b=b,
            index_of_bounds=index_of_bounds,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def get_theta_0(self):
        return self.model.theta_0

    def get_theta_0_var(self, x: torch.Tensor, y: torch.Tensor, opts: Options):
        la = self.get_laplace_approx(opts, x, y)
        covar = la.posterior_covariance
        return covar[0, 0]

    def get_laplace_approx(self, opts, x, y):
        if opts.do_fit_laplace:
            fit_laplace(
                model=self.model,
                x=x,
                y=y,
                dataset_size=opts.dataset_size,
                results_folder=opts.results_folder,
            )

        la = torch.load(os.path.join(str(opts.results_folder), "la.pt"))
        return la


class SteinScoreModel(SteinModel):
    def __init__(
        self,
        network: nn.Module,
        start_theta_0: float,
        transformation: TransformationEnum,
        const: float = 1.,
        index_of_bounds: Union[torch.Tensor, None] = None,
        a: Union[torch.Tensor, float, None] = None,
        b: Union[torch.Tensor, float, None] = None,
    ):
        super(SteinModel, self).__init__()
        self.model = SteinScoreLayer(
            network,
            start_theta_0,
            transformation,
            const=const,
            a=a,
            b=b,
            index_of_bounds=index_of_bounds,
        )
