from typing import Union

import torch
from torch import nn

from bnn_quadrature.options import ActivationEnum


def get_activation(
    act: ActivationEnum = ActivationEnum.tanh,
) -> Union[nn.ReLU, nn.Tanh, nn.CELU, nn.SELU, nn.Tanhshrink, nn.Module]:
    if act == ActivationEnum.relu:
        return nn.ReLU(inplace=True)
    if act == ActivationEnum.tanh:
        return nn.Tanh()
    if act == ActivationEnum.celu:
        return nn.CELU()
    if act == ActivationEnum.sigmoid:
        return nn.Sigmoid()
    if act == ActivationEnum.gelu:
        return nn.GELU()
    if act == ActivationEnum.silu:
        return nn.SiLU()
    if act == ActivationEnum.tanhshrink:
        return nn.Tanhshrink()
    if act == ActivationEnum.gauss:
        return Gaussian()
    raise NotImplementedError("Activation function {} not implemented".format(act))


class Gaussian(nn.Module):
    def forward(self, x: torch.Tensor):
        out = torch.exp((-(x**2)) / 2)
        return out


class NeuralNetwork(torch.nn.Module):
    """
    Builds a simple fully connected Linear Neural network
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        act: ActivationEnum = ActivationEnum.tanh,
    ):
        super(NeuralNetwork, self).__init__()
        self.activation_function = get_activation(act)
        net_list = [torch.nn.Linear(in_dim, hidden_dim), self.activation_function]
        for _ in range(num_hidden_layers):
            net_list.append(nn.Linear(hidden_dim, hidden_dim))
            net_list.append(self.activation_function)
        net_list.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*net_list)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return x
