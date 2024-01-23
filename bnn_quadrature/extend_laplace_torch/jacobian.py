from typing import List, Tuple

import torch
from backpack import extend, backpack, CTX
from backpack.extensions import BatchGrad
from laplace.curvature.backpack import _cleanup
from torch import nn
from torch.autograd import grad

from bnn_quadrature.options.device import my_device


def compute_jacobian(model, x: torch.Tensor):
    jac = calc_div_u_dp(model, x) + calc_grad_log_pi_du_dp(model, x)
    dmodel_dtheta_0 = torch.ones((x.shape[0], 1))
    jac = torch.cat((dmodel_dtheta_0, jac), dim=-1).unsqueeze(1)
    return jac.detach(), model(x).detach()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calc_div_u_dp(
    model, x: torch.Tensor
):
    nn = model.u.net
    num_params = count_parameters(nn)
    func = model._u

    batch_size = x.shape[0]
    div_u_dp = torch.zeros(num_params, batch_size)
    for d in range(x.shape[1]):
        model.zero_grad()
        x.requires_grad = True
        out = func(x)
        u_sum = torch.sum(out, dim=0).flatten()
        u_sum_flatten = u_sum.flatten()
        u_sum_d = u_sum_flatten[d]
        du_dp_d = calc_dl_dp(nn, u_sum_d)
        du_dp_dx_d = calculate_div_u_d(du_dp_d, x, d)
        div_u_dp += du_dp_dx_d.detach()

    model.zero_grad()
    CTX.remove_hooks()
    _cleanup(model)
    return div_u_dp.transpose(0, 1)


def calc_dl_dp(model: nn.Module, l: torch.Tensor):
    parameters = model.parameters()
    params = list(parameters)
    dl_dp = list(
        grad(
            l,
            params,
            create_graph=True,
            retain_graph=True,
        )
    )
    for p_i, dl_dp_i in enumerate(dl_dp):
        dl_dp[p_i] = dl_dp_i.flatten()
    dl_dp = torch.cat(dl_dp)
    return dl_dp


def calculate_div_u_d(outputs, x: torch.Tensor, i: int) -> torch.Tensor:
    """
    :param outputs: p
    :param x: B x 1 x d
    :param i: dimension along which to compute gradient
    :return:  B x p
    """

    p = outputs.shape[0]
    B = x.shape[0]
    p_step = 50
    p_list = torch.arange(0, p+p_step, p_step)
    out = torch.zeros((p, B), device=my_device.device)
    for p_l, p_h in zip(p_list[:-1], p_list[1:]):
        h_ = outputs[p_l:p_h]
        v = torch.diag(torch.ones(h_.shape[0], device=my_device.device))
        gradient = torch.autograd.grad(
            h_,
            (x,),
            grad_outputs=(v,),
            is_grads_batched=True,
            retain_graph=True
        )[0].squeeze(2)
        out[p_l:p_h] = gradient[:, :, i].detach()
    return out


def calc_grad_log_pi_du_dp(model, x):
    grad_log_pi = model.pdf.grad_log(x).detach()
    nn_simple = model.u.net
    nn_simple.output_size = x.shape[-1]
    du_dp = back_pack_jacobian(nn_simple, x.squeeze(1)).transpose(1, 2)
    du_dp = model.apply_transformation(du_dp, model.const)
    du_dp = model.apply_bounds(du_dp, x)
    grad_log_pi_du_dp = torch.einsum("ijk, ijk->ij", grad_log_pi, du_dp)
    return grad_log_pi_du_dp


def calculate_grad_log_pi_u(grad_log_pi, u, x):
    grad_pi_u = (grad_log_pi @ u(x).transpose(1, 2))
    return grad_pi_u.flatten()

"""
The following function is adapted from Laplace Version 0.1a1
( https://github.com/AlexImmer/Laplace/releases/tag/0.1a1
Copyright (c) 2021 Alex Immer, licensed under the MIT License,
cf. 3rd-party-license.txt file in the root directory of this source tree)
to use backpack to compute the Jacobian.
"""

def back_pack_jacobian(
    model: nn.Module, x: torch.Tensor
) -> torch.Tensor:
    """

    Calculates the Jacobian of the model_type with respect to the parameters at position x
    Since torch.autograd.grad expects scalar inputs, for-loop over the model_type outputs is required

    Parameters
    ----------
    model: The Jacobian is calculated of the model_type
    x: Position at which the Jacobian is calculated

    Returns
    -------
    The Jacobian of model_type, evaluation of model_type at position x - model_type(x)
    """

    model = extend(model)
    to_stack = []
    for i in range(model.output_size):
        model.zero_grad()
        out = model(x)
        with backpack(BatchGrad()):
            if model.output_size > 1:
                out[:, i].sum().backward()
            else:
                out.sum().backward()
            to_cat = []
            for param in model.parameters():
                to_cat.append(param.grad_batch.detach().reshape(x.shape[0], -1))
                delattr(param, 'grad_batch')
            Jk = torch.cat(to_cat, dim=1)
        to_stack.append(Jk)

    model.zero_grad()
    CTX.remove_hooks()
    _cleanup(model)
    if model.output_size > 1:
        return torch.stack(to_stack, dim=2).transpose(1, 2)
    else:
        return Jk.unsqueeze(-1).transpose(1, 2)
