from typing import Union

import torch
from torch import nn
from torch.nn import Parameter

from bnn_quadrature.data.pdf.pdf import PDF
from bnn_quadrature.options.device import my_device
from bnn_quadrature.options.enums import TransformationEnum


class SteinLayer(nn.Module):
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
        """
        Defines Stein layer as defined in the Paper
        :param network: Neural network u
        :param pdf: partial density function
        :param start_theta_0: Starting value for the scalar theta_0, will be optimized during training
        """
        super(SteinLayer, self).__init__()
        self.m = transformation
        self.const = const
        self.theta_0 = Parameter(
            torch.tensor([start_theta_0])
        )
        self.pdf = pdf
        self.u = network
        self.index_of_bounds, self.a, self.b = self.init_bounds(a, b, index_of_bounds)

    @staticmethod
    def init_bounds(a, b, index_of_bounds):
        if index_of_bounds is None and (a is not None or b is not None):
            raise TypeError(
                f"Please provide an index for the dimension you want to apply the integration boundaries to"
            )
        if isinstance(a, float) or isinstance(b, float):
            if len(index_of_bounds) != 1:
                raise TypeError(
                    f"a or b need to be of type Tensor for index_of_bounds list with length {len(index_of_bounds)}"
                )
        if isinstance(a, torch.Tensor):
            a = a.to(my_device.device)
        if isinstance(b, torch.Tensor):
            b = b.to(my_device.device)
        return index_of_bounds, a, b

    def _apply(self, fn):
        super(SteinLayer, self)._apply(fn)
        if isinstance(self.a, torch.Tensor):
            self.a = fn(self.a)
        if isinstance(self.b, torch.Tensor):
            self.b = fn(self.b)

    def _u(self, x: torch.Tensor):
        out = self.u(x)

        out = self.apply_transformation(out, self.const)

        out = self.apply_bounds(out, x)

        return out

    def apply_bounds(self, out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        o = torch.ones_like(x)
        if self.a is not None:
            o[:, :, self.index_of_bounds] = x[:, :, self.index_of_bounds] - self.a
        if self.b is not None:
            o[:, :, self.index_of_bounds] = o[:, :, self.index_of_bounds] * (self.b - x[:, :, self.index_of_bounds])
        out = out * o
        return out

    def apply_transformation(self, out: torch.Tensor, const: float) -> torch.Tensor:
        """
        :param out:
        :param const:
        :return:
        """
        if self.m == TransformationEnum.none:
            return out
        if self.m == TransformationEnum.const:
            return out * 1/const
        else:
            raise KeyError(f"{self.m} is not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Tensor of shape B x 1 x D where B ist the Batch size and D is the dimension
        :return: Tensor of shape B X 1
        """
        div_u = calculate_div_u(self._u, x)
        grad_log_pi_u = calculate_grad_log_pi_u(self.pdf.grad_log(x), self._u, x)
        out = (grad_log_pi_u + div_u + self.theta_0).unsqueeze(-1)
        return out


class SteinScoreLayer(SteinLayer):
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
        """
        Defines Stein layer as defined in the Paper
        :param network: Neural network u
        :param pdf: partial density function
        :param start_theta_0: Starting value for the scalar theta_0, will be optimized during training
        """
        super(SteinLayer, self).__init__()
        self.m = transformation
        self.theta_0 = Parameter(
            torch.tensor([start_theta_0])
        )
        self.const = const
        self.u = network
        self.index_of_bounds, self.a, self.b = self.init_bounds(a, b, index_of_bounds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        :param x: Tensor of shape B x 1 x D where B ist the Batch size and D is the dimension
        :return: Tensor of shape B X 1
        """
        x, scores = torch.chunk(x, 2, dim=-1)
        div_u = calculate_div_u(self._u, x)
        grad_log_pi_u = calculate_grad_log_pi_u(scores, self._u, x)
        out = (grad_log_pi_u + div_u + self.theta_0).unsqueeze(-1)
        return out


def calculate_grad_log_pi_u(grad_log_pi, u, x):
    grad_pi_u = (grad_log_pi @ u(x).transpose(1, 2))
    return grad_pi_u.flatten()


def calculate_div_u(u, x: torch.Tensor) -> torch.Tensor:
    """

    :param u: function from B x 1 x D -> B x 1 x D
    :param x: B x 1 x D
    :return:  B
    """
    with torch.enable_grad():
        x.requires_grad = True
        outputs = u(x)
        div_du_dx = torch.zeros(
            x.shape[0],
            1,
            device=my_device.device
        )
        for i in range(x.shape[-1]):
            v = torch.ones(x.shape[0], 1, device=my_device.device)
            gradient = torch.autograd.grad(
                outputs[:, :, i],
                x,
                grad_outputs=v,
                create_graph=True,
            )[0]
            div_du_dx += gradient[:, :, i]
    div_u = div_du_dx.squeeze(1)
    return div_u
